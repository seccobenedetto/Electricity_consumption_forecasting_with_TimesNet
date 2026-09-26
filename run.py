import argparse
import random
import time
import numpy as np
import torch
import os

from exp_pipeline import Exp_Long_Term_Forecast
from data_loader import data_provider


def main(args):

    # ----------------------
    # Experiment
    # ----------------------
    Exp = Exp_Long_Term_Forecast

    exp = Exp(args)

    # this is essentially a string that describes the experiment configuration, used for saving checkpoints and logs
    setting = ( 
        f"sl{args.seq_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_el{args.e_layers}_"
        f"tk{args.top_k}"
    )

    if args.train:
        print("\n================ TRAINING ================\n")
        exp.train(setting)

    if args.test:
        print("\n================ TESTING ================\n")
        exp.test(setting, load_best=True)

    if args.predict_samples:
        print("\n================ PREDICTING SINGLE SAMPLES ================\n")
        exp.predict_samples(setting, num_batches=1, load_best=True)





if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description="TimesNet - Electricity Forecasting")

    # ======================
    # ACTION
    # ======================
    parser.add_argument("--train", action="store_true", help="Train the model")  # whether to train the model
    parser.add_argument("--test", action="store_true", help="Test the model")    # whether to test the model (after loading the best checkpoint)
    parser.add_argument("--predict_samples", action="store_true", help="Predict single samples")  # whether to predict single samples (for plotting purposes)

    # ======================
    # DATA
    # ======================
    parser.add_argument("--root_path", type=str, default="../electricity_dataset/")
    parser.add_argument("--data_path", type=str, default="electricity.csv")

    parser.add_argument("--seq_len", type=int, default=96)    # length of the input sequence that the model uses to make predictions.
    parser.add_argument("--label_len", type=int, default=48)  # length of the label sequence that the model uses for training.
    parser.add_argument("--pred_len", type=int, default=96)   # prediction length, i.e., how many future time steps the model should predict.

    # ======================
    # MODEL
    # ======================
    #parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--e_layers", type=int, default=2)     # number of sequential TimesBlocks stacked in the model

    parser.add_argument("--enc_in", type=int, default=321)     # number of initial input features/channels for the network
    parser.add_argument("--d_model", type=int, default=64)     # hidden dimension of the model
    parser.add_argument("--c_out", type=int, default=321)      # number of final output features/channels for the network

    parser.add_argument("--d_ff", type=int, default=128)       # hidden dimension of the feedforward network in TimesBlock
    parser.add_argument("--num_kernels", type=int, default=6)  # number of convolutional kernels used in the TimesBlock for capturing different temporal patterns
    parser.add_argument("--top_k", type=int, default=5)        # number of top periods to consider in the TimesBlock

    # ======================
    # TRAINING
    # ======================
    parser.add_argument("--num_workers", type=int, default=4)  # number of subprocesses on the CPU to use for data loading. 0 means that the data will be loaded in the main process.
    parser.add_argument("--batch_size", type=int, default=32)  # number of samples per batch to load. A larger batch size can speed up training but requires more memory.

    parser.add_argument("--learning_rate", type=float, default=1e-4) # learning rate for the Adam optimizer
    parser.add_argument("--lradj", type=str, default="type1")        # learning rate adjustment strategy
    parser.add_argument("--train_epochs", type=int, default=10)      # number of epochs to train the model. An epoch is one complete pass through the entire training dataset.
    parser.add_argument("--patience", type=int, default=3)           # number of epochs with no improvement after which training will be stopped early (EarlyStopping)

    parser.add_argument("--use_gpu", type=bool, default=True)        # whether to use GPU for training. If True and a GPU is available, the model will be trained on the GPU; otherwise, it will be trained on the CPU.
    parser.add_argument("--gpu", type=int, default=0)

    # ======================
    # CHECKPOINTS
    # ======================
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")  # directory where model checkpoints will be saved during training

    args = parser.parse_args()

    print("Args:", args)

    main(args)

    # debug:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTOTAL TIME TAKEN: {elapsed_time:.2f} seconds")