#!/usr/bin/env python3
"""Build fixed-KV decode variants from the optimized fixed-cache graph.

Only the cache-capacity dimension and the helper index initializer are changed;
all operators, weights, quantization, and graph ordering are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


KV_PREFIXES = ("past_key_", "past_value_", "present_key_", "present_value_")


def _dim_value(dim: onnx.TensorShapeProto.Dimension) -> int | None:
    return int(dim.dim_value) if dim.HasField("dim_value") else None


def _set_capacity(value_info: onnx.ValueInfoProto, capacity: int) -> None:
    shape = value_info.type.tensor_type.shape
    if len(shape.dim) != 4 or _dim_value(shape.dim[1]) is None:
        raise ValueError(f"unexpected KV shape for {value_info.name}")
    shape.dim[1].dim_value = capacity
    shape.dim[1].ClearField("dim_param")


def build_variant(source: Path, output: Path, capacity: int) -> None:
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    model = onnx.load(str(source), load_external_data=False)
    kv_values = [
        value_info
        for value_info in (*model.graph.input, *model.graph.output)
        if value_info.name.startswith(KV_PREFIXES)
    ]
    if len(kv_values) != 48:
        raise ValueError(f"expected 48 KV graph values, found {len(kv_values)}")

    source_capacities = {_dim_value(value_info.type.tensor_type.shape.dim[1]) for value_info in kv_values}
    if len(source_capacities) != 1 or None in source_capacities:
        raise ValueError(f"KV capacities are not uniform: {source_capacities}")
    source_capacity = next(iter(source_capacities))

    fixed_indices = next(
        (initializer for initializer in model.graph.initializer if initializer.name == "fixed_kv_indices"),
        None,
    )
    if fixed_indices is None:
        raise ValueError("source graph has no fixed_kv_indices initializer")
    old_indices = numpy_helper.to_array(fixed_indices)
    if old_indices.shape != (source_capacity,) or not np.array_equal(
        old_indices, np.arange(source_capacity, dtype=np.int32)
    ):
        raise ValueError("fixed_kv_indices is not the expected contiguous int32 range")

    for value_info in kv_values:
        _set_capacity(value_info, capacity)

    fixed_indices.CopyFrom(
        numpy_helper.from_array(np.arange(capacity, dtype=np.int32), name="fixed_kv_indices")
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output), save_as_external_data=False)
    print(f"wrote {output} (source_capacity={source_capacity}, capacity={capacity})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, action="append", required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source graph not found: {source}")
    for capacity in args.capacity:
        output = args.output_dir.expanduser().resolve() / f"moss_tts_decode_step.inplace-fixed{capacity}.pruned.onnx"
        build_variant(source, output, capacity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
