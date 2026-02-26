#!/usr/bin/env bash
set -e

CKPT="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt"

EXPLAIN="python3 explain_HFSemiClassifier_bm.py"

# ===== pT bins =====
PT_BINS=(
  "3.0 4.0"
  "4.0 5.0"
  "5.0 6.0"
  "6.0 8.0"
  "8.0 10.0"
)

OUTBASE="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Replot/ptscan2/"

mkdir -p "${OUTBASE}"

for BIN in "${PT_BINS[@]}"; do
  read PTMIN PTMAX <<< "${BIN}"

  echo "========================================"
  echo " Running explain for pT in [${PTMIN}, ${PTMAX})"
  echo "========================================"

  OUTPREFIX="${OUTBASE}/pt${PTMIN}-${PTMAX}"

  ${EXPLAIN} \
    --ckpt "${CKPT}" \
    --pt-min "${PTMIN}" \
    --pt-max "${PTMAX}" \
    --balance-ds \
    --out-prefix "${OUTPREFIX}"

done

echo "All pT bins done."
