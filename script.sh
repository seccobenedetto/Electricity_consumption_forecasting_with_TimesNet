#!/bin/bash

# =========================
# Hyperparameter grid
# =========================

PRED_LENS=(96 192 336)
SEQ_LENS=(96 192)

mkdir -p logs

# =========================
# Run experiments
# =========================

for seq_len in "${SEQ_LENS[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do

    echo "======================================"
    echo "Running TimesNet | seq_len=$seq_len | pred_len=$pred_len"
    echo "======================================"

    python -u run.py \
      --predict_samples \
      --model TimesNet \
      --d_model 128 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      #> logs/TimesNet_${seq_len}_${pred_len}.log 2>&1
   
    sleep 2
  done
done

echo "All experiments completed."
