#!/bin/bash

# 出错就退出
set -e

# 输出文件名你可以按需要改
OUT0="ppHF_eXDecay_5B_1.root"
OUT1="ppHF_eXDecay_5B_2.root"

# 合并 0-499
echo "Merging 0-499 -> ${OUT0}"
hadd -f "${OUT0}" ppHF_eXDecay_10M_{0..499}.root

# 合并 500-999
echo "Merging 500-999 -> ${OUT1}"
hadd -f "${OUT1}" ppHF_eXDecay_10M_{500..999}.root

echo "Done."
