from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def test_control_reader_drains_coalesced_jsonl_commands(tmp_path: Path):
    """One pipe write may contain FINAL_RESULT and STOP; consume both now."""
    source = tmp_path / "control_protocol_test.cpp"
    binary = tmp_path / "control_protocol_test"
    source.write_text(
        textwrap.dedent(
            r"""
            #include "vision/yolov8_adjudicator/src/control_protocol.h"

            #include <unistd.h>

            #include <string>

            int main() {
                int fds[2];
                if (::pipe(fds) != 0) return 10;
                const std::string payload =
                    "{\"command\":\"FINAL_RESULT\"}\n"
                    "{\"command\":\"STOP_ADJUDICATION\"}\n";
                if (::write(fds[1], payload.data(), payload.size()) !=
                    static_cast<ssize_t>(payload.size())) return 11;

                vision_control::CommandReader reader;
                const auto commands = reader.read_ready(fds[0], 100);
                ::close(fds[0]);
                ::close(fds[1]);

                if (commands.size() != 2) return 20;
                if (commands[0].find("FINAL_RESULT") == std::string::npos) return 21;
                if (commands[1].find("STOP_ADJUDICATION") == std::string::npos) return 22;
                return 0;
            }
            """
        ),
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert run_result.returncode == 0, run_result.stderr
