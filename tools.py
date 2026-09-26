import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    Here we adjust the learning rate according to the epoch number:
    - type1: the learning rate is halved every epoch
    - type2: the learning rate is set to a specific value at certain epochs
    - type3: the learning rate is kept constant for the first 3 epochs, then decays by 0.9 every epoch
    - cosine: the learning rate follows a cosine schedule, starting from the initial learning rate and decaying to 0 over the course of training
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience   # how many times will you tolerate for loss not being on decrease
        self.verbose = verbose     # whether to print tip info
        self.counter = 0           # now how many times loss not on decrease
        self.best_score = None     # best validation performance seen so far
        self.early_stop = False    # flag: should we stop training?
        self.val_loss_min = np.inf # best validation loss (for logging + checkpoint message)
        self.delta = delta         # minimum improvement required to count as “better”

    def __call__(self, val_loss, model, path):
        score = -val_loss  # convert loss to a score
        
        # ================== FIRST EPOCH CASE ======================
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        # ================== NOT IMPROVEMENT CASE ======================
        # meaning: current score is not 'delta' better than best_score, representing that 
        # further training may not bring remarkable improvement in loss. 
        elif score < self.best_score + self.delta:  # = NOT good enough improvement
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 'No Improvement' times become higher than patience --> Stop Further Training
            if self.counter >= self.patience:
                self.early_stop = True

        # ================== IMPROVEMENT CASE ======================
        else: # model's loss is still on decrease, save the now best model and go on training
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
    # used for saving the current best model
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    """
    'model.state_dict()' is a Python dictionary object that maps each layer to its parameter tensor.
    It is used to save the model's learned parameters (weights and biases) to a file.
    Example of dict:
    {
        "layer1.weight": tensor(...),
        "layer1.bias": tensor(...),
        ...
    }
    """


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
