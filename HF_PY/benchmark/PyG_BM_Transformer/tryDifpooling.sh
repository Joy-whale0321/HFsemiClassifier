#!/usr/bin/env bash

set -e

POOLINGS=("sum" "mean" "max" "attn" "attn_mean")

for POOL in "${POOLINGS[@]}"; do
    python3 train_HFSemiClassifier.py \
        --pooling "${POOL}" 

    echo "Finished pooling = ${POOL}"
    echo
done
