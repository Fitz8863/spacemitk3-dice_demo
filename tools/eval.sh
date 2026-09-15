#!/usr/bin/env bash
# 一键回归：合成图自检 + 真实样本帧批量检测
#
#   ./tools/eval.sh              # 默认跑 samples/ + --self-test
#   ./tools/eval.sh <图片目录>    # 换成自己的目录（例如生产采集的数据集）
#
# 退出码：0 = 自检全过；非 0 表示有回归。
#
# 注意：本脚本一律带 --no-rtsp，避免回归测试顺手把推流拉起来。

set -u
cd "$(dirname "$0")/.." || exit 1

BIN=./build/circle_detect
# 关掉一切对外副作用：只做识别
OPTS=(--no-rtsp)
DIR=${1:-samples}

if [[ ! -x $BIN ]]; then
  echo "[err] 找不到 $BIN，先编译："
  echo "      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
  exit 1
fi

echo "=============================================================="
echo " 1) 合成图自检（无需摄像头）"
echo "=============================================================="
"$BIN" "${OPTS[@]}" --self-test
SELF=$?

echo
echo "=============================================================="
echo " 2) 真实帧批量检测：$DIR"
echo "=============================================================="
if [[ ! -e $DIR ]]; then
  echo "[skip] $DIR 不存在"
  exit $SELF
fi

"$BIN" "${OPTS[@]}" --image "$DIR" --expected 2 --quiet --summary 2>/dev/null | tail -1

echo
echo "  不同工作分辨率的耗时/命中率："
for w in 320 480 640 960; do
  printf "    work-width %4s : " "$w"
  "$BIN" "${OPTS[@]}" --image "$DIR" --expected 2 --method mask \
      --work-width "$w" --summary 2>/dev/null | tail -1
done

echo
if [[ $SELF -eq 0 ]]; then
  echo "[OK] 自检通过"
else
  echo "[FAIL] 自检存在失败项"
fi
exit $SELF
