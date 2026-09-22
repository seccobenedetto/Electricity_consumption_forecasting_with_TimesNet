import argparse
import random
import numpy as np
import torch
import os

from exp_pipeline import Exp_Long_Term_Forecast
from data_loader import data_provider


def main(args):

    # ----------------------
    # GPU setup
    # ----------------------
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.device = device

    # ----------------------
    # Experiment
    # ----------------------
    Exp = Exp_Long_Term_Forecast

    exp = Exp(args)

    setting = (
        f"{args.model}_"
        f"sl{args.seq_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_el{args.e_layers}_"
        f"tk{args.top_k}"
    )

    print("\n================ TRAINING ================\n")
    exp.train(setting)

    print("\n================ TESTING ================\n")
    exp.test(setting)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TimesNet - Electricity Forecasting")

    # ======================
    # DATA
    # ======================
    parser.add_argument("--root_path", type=str, default="../electricity_dataset/")
    parser.add_argument("--data_path", type=str, default="electricity.csv")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)

    # ======================
    # MODEL
    # ======================
    parser.add_argument("--model", type=str, default="TimesNet")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--e_layers", type=int, default=2)

    parser.add_argument("--enc_in", type=int, default=321)
    parser.add_argument("--c_out", type=int, default=321)

    # ======================
    # TRAINING
    # ======================
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)

    # ======================
    # CHECKPOINTS
    # ======================
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    args = parser.parse_args()

    print("Args:", args)

    main(args)