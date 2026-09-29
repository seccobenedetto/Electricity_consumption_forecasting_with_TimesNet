#!/bin/bash

# =========================
# Hyperparameter grid
# =========================

# PRED_LENS=(96 192 336)
# SEQ_LENS=(96 192)
K_VALUES=( 1 2 3 4 5 6 7 )

mkdir -p logs

# =========================
# Run experiments
# =========================

for k_idx in "${K_VALUES[@]}"; do
  echo "======================================"
  echo "Running TimesNet | k_idx=$k_idx"
  echo "======================================"

  python -u run.py \
    --train \
    --test \
    --model TimesNet \
    --top_k "$k_idx" \
    --d_model 128 \
    --seq_len 96 \
    --pred_len 96 2>&1 | tee "logs/TimesNet_k${k_idx}.log"

  sleep 5
done

echo "All experiments completed."
