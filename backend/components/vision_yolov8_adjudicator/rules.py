"""Game-agnostic rule evaluation for visual adjudication profiles.

The runtime only supplies observations.  This module is the small, pure
translation layer that turns those observations into an outcome and wraps it
in the stable result contract consumed by the job/SSE layer.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
import math
from typing import Any


class RuleError(ValueError):
    """Raised when an observation, rule, or adjudication decision is invalid."""


def fuse_yolo_outcomes(outcomes: Sequence[str]) -> str | None:
    """Return the strict-majority outcome, or ``None`` for a tie/no votes.

    Every vote participates in the denominator.  In particular, an even
    number of views split evenly and must never be turned into an arbitrary
    winner.
    """

    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        raise RuleError("outcomes must be a sequence of strings")
    if not outcomes:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in outcomes):
        raise RuleError("outcomes must contain non-empty strings")
    counts = Counter(value.strip() for value in outcomes)
    winner, votes = counts.most_common(1)[0]
    return winner if votes * 2 > len(outcomes) else None


def _participants(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    values = observation.get("participants")
    if not isinstance(values, Mapping) or len(values) != 2:
        raise RuleError("observation.participants must contain exactly two participants")
    return values


def _numeric_value(value: Any, participant: str) -> float:
    # A participant may be represented by one class value or a list of
    # detected values (for example the five dice on one side).
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise RuleError(f"participant {participant!r} has no numeric values")
    parsed: list[float] = []
    for item in values:
        if isinstance(item, bool):
            raise RuleError(f"participant {participant!r} values must be numbers")
        if isinstance(item, Real):
            number = float(item)
        elif isinstance(item, str):
            try:
                number = float(item.strip())
            except ValueError as exc:
                raise RuleError(f"participant {participant!r} values must be numbers") from exc
        else:
            raise RuleError(f"participant {participant!r} values must be numbers")
        if not math.isfinite(number):
            raise RuleError(f"participant {participant!r} values must be finite")
        parsed.append(number)
    return float(sum(parsed))


def _numeric_compare(rule: Mapping[str, Any], observation: Mapping[str, Any]) -> str:
    participants = _participants(observation)
    expected_count = rule.get("expected_count")
    if expected_count is not None:
        if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0:
            raise RuleError("numeric_compare expected_count must be a positive integer")
        for participant, raw in participants.items():
            values = raw if isinstance(raw, (list, tuple)) else [raw]
            if len(values) != expected_count:
                raise RuleError(f"participant {participant!r} expected_count is {expected_count}")
    if rule.get("aggregation", "sum") != "sum":
        raise RuleError("numeric_compare currently supports aggregation=sum only")
    names = list(participants)
    left_name, right_name = names[0], names[1]
    left = _numeric_value(participants[left_name], left_name)
    right = _numeric_value(participants[right_name], right_name)
    tie_value = rule.get("tie_value", "TIE")
    if not isinstance(tie_value, str) or not tie_value:
        raise RuleError("numeric_compare tie_value must be a non-empty string")
    if left == right:
        return tie_value
    higher_wins = rule.get("higher_wins", True)
    if not isinstance(higher_wins, bool):
        raise RuleError("numeric_compare higher_wins must be boolean")
    higher_name = left_name if left > right else right_name
    lower_name = right_name if left > right else left_name
    return higher_name if higher_wins else lower_name


def _categorical_compare(rule: Mapping[str, Any], observation: Mapping[str, Any]) -> str:
    participants = _participants(observation)
    relations = rule.get("relations")
    if not isinstance(relations, Mapping) or not relations:
        raise RuleError("categorical_relation relations must be a non-empty object")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in relations.items()):
        raise RuleError("categorical_relation relations must map strings to strings")
    names = list(participants)
    left_name, right_name = names[0], names[1]
    left, right = participants[left_name], participants[right_name]
    if not isinstance(left, str) or not isinstance(right, str):
        raise RuleError("categorical_relation participant values must be strings")
    categories = set(relations) | set(relations.values())
    if left not in categories or right not in categories:
        raise RuleError("unknown categorical outcome")
    tie_value = rule.get("tie_value", "TIE")
    if not isinstance(tie_value, str) or not tie_value:
        raise RuleError("categorical_relation tie_value must be a non-empty string")
    if left == right:
        return tie_value
    if relations.get(left) == right:
        return left_name
    if relations.get(right) == left:
        return right_name
    # Distinct categories with no declared relation are not a valid winner;
    # returning an implicit tie would hide a malformed game profile.
    raise RuleError(f"no categorical relation between {left!r} and {right!r}")


def evaluate_rule(rule: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> str:
    """Evaluate a declared profile rule over one or more view observations."""

    if not isinstance(rule, Mapping):
        raise RuleError("rule must be an object")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)) or not observations:
        raise RuleError("observations must be a non-empty sequence")
    if any(not isinstance(item, Mapping) for item in observations):
        raise RuleError("observations must contain objects")
    kind = rule.get("kind")
    if kind == "numeric_compare":
        values = [_numeric_compare(rule, item) for item in observations]
    elif kind == "categorical_relation":
        values = [_categorical_compare(rule, item) for item in observations]
    else:
        raise RuleError(f"unsupported rule kind: {kind!r}")
    if len(values) == 1:
        return values[0]
    outcome = fuse_yolo_outcomes(values)
    if outcome is None:
        raise RuleError("no strict majority across observations")
    return outcome


def finalize_outcome(
    *, yolo_outcome: str | None, llm_outcome: str | None, llm_status: str
) -> dict[str, Any]:
    """Apply the documented YOLO/LLM precedence and return decision metadata."""

    if llm_status not in {"success", "timeout", "disabled", "failure", "error"}:
        raise RuleError(f"unsupported LLM status: {llm_status!r}")
    if yolo_outcome is not None and (not isinstance(yolo_outcome, str) or not yolo_outcome.strip()):
        raise RuleError("yolo_outcome must be a non-empty string or None")
    if llm_outcome is not None and (not isinstance(llm_outcome, str) or not llm_outcome.strip()):
        raise RuleError("llm_outcome must be a non-empty string or None")

    if llm_status == "disabled":
        if yolo_outcome is None:
            raise RuleError("disabled LLM requires a YOLO outcome")
        value = yolo_outcome.strip()
        source = "yolo_only"
        verification_status = "disabled"
    elif llm_status == "success":
        if llm_outcome is None:
            raise RuleError("LLM success requires an outcome")
        value = llm_outcome.strip()
        source = "consensus" if yolo_outcome == value else "llm_override"
        verification_status = "agreed" if source == "consensus" else "overridden"
    elif llm_status == "timeout":
        if yolo_outcome is None:
            raise RuleError("LLM timeout cannot be used without a YOLO outcome")
        value = yolo_outcome.strip()
        source = "yolo_timeout_fallback"
        verification_status = "timeout_fallback"
    else:
        if yolo_outcome is None:
            raise RuleError("LLM failure cannot be used without a YOLO outcome")
        value = yolo_outcome.strip()
        source = "yolo_failure_fallback"
        verification_status = "failure_fallback"

    return {
        "adjudicated": True,
        "outcome": {"kind": "winner", "value": value},
        "decision_source": source,
        "verification": {
            "status": verification_status,
            "yolo_outcome": yolo_outcome,
            "llm_outcome": llm_outcome,
            "llm_called": llm_status not in {"disabled"},
        },
    }


def project_result(
    profile: Mapping[str, Any], decision: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach profile/evidence and migration-only dice fields to a decision."""

    if not isinstance(profile, Mapping) or not isinstance(decision, Mapping) or not isinstance(evidence, Mapping):
        raise RuleError("profile, decision and evidence must be objects")
    profile_id = profile.get("game_id") or profile.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        raise RuleError("profile.game_id must be a non-empty string")
    outcome = decision.get("outcome")
    if not isinstance(outcome, Mapping) or not isinstance(outcome.get("value"), str):
        raise RuleError("decision.outcome.value must be a string")
    allowed = profile.get("llm", {}).get("allowed_outcomes") if isinstance(profile.get("llm"), Mapping) else None
    if isinstance(allowed, Sequence) and outcome["value"] not in allowed:
        raise RuleError("outcome is not listed in profile llm.allowed_outcomes")

    result = dict(decision)
    result["profile_id"] = profile_id
    result["provider_id"] = "vision_yolov8_adjudicator"
    result["evidence"] = dict(evidence)

    # Keep old dice clients working without making these fields part of the
    # generic contract.  Other games simply receive the generic result.
    participants = evidence.get("participants")
    if isinstance(participants, Mapping):
        for name, prefix in (("LEFT", "left"), ("RIGHT", "right")):
            raw = participants.get(name)
            if isinstance(raw, (list, tuple)):
                values = list(raw)
                result[f"{prefix}_values"] = values
                if values and all(isinstance(item, Real) and not isinstance(item, bool) for item in values):
                    result[f"{prefix}_sum"] = sum(values)
        if profile_id == "dice" and isinstance(result.get("outcome"), Mapping):
            result["winner"] = result["outcome"]["value"]
            result["first_dice"] = list(participants.get("LEFT", [])) if isinstance(participants.get("LEFT"), (list, tuple)) else []
            result["second_dice"] = list(participants.get("RIGHT", [])) if isinstance(participants.get("RIGHT"), (list, tuple)) else []
            result["first_sum"] = result.get("left_sum", 0)
            result["second_sum"] = result.get("right_sum", 0)
            result["source"] = result.get("decision_source", "provider")
            verification = result.get("verification")
            if isinstance(verification, Mapping):
                result["llm_winner"] = verification.get("llm_outcome")
    return result
