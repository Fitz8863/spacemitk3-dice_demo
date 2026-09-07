#!/usr/bin/env bash
set -euo pipefail

module_dir="components/model_zoo/asr"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/asr-pr-test.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

g++ -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter \
  "${module_dir}/tests/asr_pr_contract_test.cpp" \
  "${module_dir}/src/asr_presets.cpp" \
  -I"${module_dir}/include" \
  -I"${module_dir}/src" \
  -o "${build_dir}/asr_pr_contract_test"

"${build_dir}/asr_pr_contract_test" --invalid-config-error-path
