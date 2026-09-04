#!/usr/bin/env bash
set -euo pipefail

module_dir="components/model_zoo/asr"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/asr-gemma4-test.XXXXXX")"
server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -rf "${build_dir}"
}
trap cleanup EXIT

pkg_config_flags="$(pkg-config --cflags --libs libcurl sndfile)"
read -r -a flags <<< "${pkg_config_flags}"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter \
  "${module_dir}/tests/gemma4_asr_backend_contract_test.cpp" \
  "${module_dir}/src/backends/gemma4_asr/gemma4_asr_backend.cpp" \
  "${module_dir}/src/backends/llama_audio/llama_audio_client.cpp" \
  -I"${module_dir}/src" \
  "${flags[@]}" \
  -pthread \
  -o "${build_dir}/gemma4_asr_backend_contract_test"

python3 "${module_dir}/tests/gemma4_asr_mock_server.py" \
  --port-file "${build_dir}/port" \
  >"${build_dir}/server.log" 2>&1 &
server_pid=$!

for _ in {1..100}; do
  if [[ -s "${build_dir}/port" ]]; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    cat "${build_dir}/server.log"
    exit 1
  fi
  sleep 0.05
done

if [[ ! -s "${build_dir}/port" ]]; then
  echo "mock Gemma4 ASR server did not start" >&2
  cat "${build_dir}/server.log"
  exit 1
fi

endpoint="http://127.0.0.1:$(cat "${build_dir}/port")/v1/audio/transcriptions"
if ! "${build_dir}/gemma4_asr_backend_contract_test" \
    "${endpoint}" "${build_dir}/input.wav"; then
  cat "${build_dir}/server.log"
  exit 1
fi
