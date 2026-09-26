import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# custom imports:
from data_loader import data_provider
from tools import EarlyStopping, adjust_learning_rate
from metrics import metric
from TimesNet import Model as TimesNet


class Exp_Basic(object):
    """
    This is a minimal base class for experiments. 
    It handles device setup, model construction, and provides hooks for  training and testing.

    Minimal base class:
    - device setup
    - model construction hook
    """

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device() # acquire the device (CPU or GPU) based on args
        self.model = self._build_model().to(self.device) # build the model and move it to the device

    # this function is meant to be overridden by subclasses to build the specific model for the experiment.
    def _build_model(self):
        raise NotImplementedError

    # this function acquires the device (CPU or GPU) based on args
    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"[Device] CUDA:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("[Device] CPU")
        return device


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    Simplified forecasting pipeline:
    - NO time encoding (no x_mark / y_mark)
    - Electricity dataset only
    - Pure value-based encoder-decoder forecasting
    """

    def __init__(self, args):
        super().__init__(args)

    # -------------------------------------------------------
    # MODEL
    # -------------------------------------------------------
    def _build_model(self):

        model = TimesNet(self.args).float()

        return model

    # -------------------------------------------------------
    # DATA
    # -------------------------------------------------------
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    # -------------------------------------------------------
    # OPTIM + LOSS
    # -------------------------------------------------------
    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    # -------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------
    # this function evaluates the model on the validation/test set and returns the average loss.
    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                # here we don't use time embeddings, so we just pass the raw sequences to the model.

                # ------------------------------------------------
                # batch_x: [B, seq_len, 321]
                # batch_y: [B, label_len + pred_len, 321]
                # ------------------------------------------------

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                B, L, C = batch_y.shape

                # forward WITHOUT time embeddings
                # outputs: [B, pred_len, 321]
                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                true =    batch_y[:, -self.args.pred_len:, :]

                loss = criterion(outputs, true)
                total_loss.append(loss.item())

        self.model.train()

        return np.mean(total_loss)

    # -------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------
    def train(self, setting):

        # ================== DATA LOADER ======================
        # with position [0] we get the dataset, with [1] we get the DataLoader
        train_loader = self._get_data('train')[1]
        vali_loader = self._get_data('val')[1]
        test_loader = self._get_data('test')[1]

        # ================== OPTIMIZER AND LOSS ======================
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        train_history = {   # this dictionary will store the training, validation, and test losses for each epoch
            "train_loss": [],
            "val_loss": [],
            "test_loss": []
        }

        # ================== EARLY STOPPING ======================
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        # ================== EPOCH TRAINING ======================
        for epoch in range(self.args.train_epochs):

            self.model.train()
            train_losses = []
            epoch_time = time.time() # start time of the epoch
            #iter_count = 0           # count the number of iterations in this epoch

            # begin training in this epoch:
            # loop over batches (each iteration = one gradient update)
            for i, (batch_x, batch_y) in enumerate(train_loader):

                #iter_count += 1
                # ------------------------------------------------
                # batch_x: [B, seq_len, 321]
                # batch_y: [B, label_len + pred_len, 321]
                # ------------------------------------------------

                # Move to GPU:
                batch_x = batch_x.float().to(self.device) # input sequence
                batch_y = batch_y.float().to(self.device) # target sequence (known labels + future zeros)

                optimizer.zero_grad()

                # ================== FORWARD PASS ======================
                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]  # we only care about the last pred_len time steps for loss computation
                true =    batch_y[:, -self.args.pred_len:, :]  # the true values for the last pred_len time steps

                # ================== LOSS COMPUTATION ======================
                loss = criterion(outputs, true)
                train_losses.append(loss.item())

                # ================== BACKPROP AND UPDATE ======================
                loss.backward()
                optimizer.step()

                if i % 100 == 0 and i != 0:
                    elapsed = time.time() - time_now
                    speed = elapsed / 100

                    remaining_batches = (self.args.train_epochs - epoch - 1) * train_steps + (train_steps - i)
                    left_time = speed * remaining_batches

                    print(f"Epoch {epoch} | Iter {i} | Loss {loss.item():.6f}   |   "
                          f"Speed {speed:.1f}s/iter | left estimated time {left_time/60:.1f} min"  )

                    time_now = time.time()
            # ================== SINGLE EPOCH (=all batches) ENDS ======================

            # now we evaluate the model on the validation and test sets after this epoch of training:
            vali_loss = self.vali(vali_loader, criterion)  # to detect overfitting and to detect early stopping
            test_loss = self.vali(test_loader, criterion)  # just for sanity check, we don't use test loss for early stopping
            train_loss = np.mean(train_losses)

            print(
                f"    [Epoch {epoch}] "
                f"Time cost: {time.time() - epoch_time:.1f}s, "
                f"train_loss={train_loss:.6f} "
                f"vali_loss={vali_loss:.6f} "
                f"test_loss={test_loss:.6f}"
            )

            train_history["train_loss"].append(train_loss)
            train_history["val_loss"].append(vali_loss)
            train_history["test_loss"].append(test_loss)

            # Decide whether to trigger Early Stopping. if early_stop is true, it means that 
            # this epoch's training is now at a flat slope, so stop further training for this epoch.
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)
        # ================== ALL EPOCHS END ======================

        np.save(os.path.join(path, "loss_history.npy"), train_history) # save the training history for later analysis

        # here we load the best model saved during training (choosen inside EarlyStopping) 
        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth")))

        return self.model
    

    # -------------------------------------------------------
    # TEST
    # -------------------------------------------------------
    def test(self, setting, load_best=True):

        test_loader = self._get_data('test')[1]

        # -------------------------------------------------------
        # LOAD BEST MODEL IF REQUESTED
        # -------------------------------------------------------
        if load_best:
            path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            print(f"Loading best model from: {path}")
            self.model.load_state_dict(torch.load(path, map_location=self.device))

        self.model.eval()

        preds, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x) # forward pass to get predictions

                outputs = outputs[:, -self.args.pred_len:, :]
                true =    batch_y[:, -self.args.pred_len:, :]

                preds.append(outputs.cpu().numpy())
                trues.append(true.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print("Test shapes:", preds.shape, trues.shape)

        mse, mae = metric(preds, trues)

        print(f"MSE={mse:.6f}, MAE={mae:.6f}")

        return mse, mae



    # -------------------------------------------------------
    # PREDICT SINGLE SAMPLES and SAVE RESULTS for plotting
    # -------------------------------------------------------
    def predict_samples(self, setting, num_batches=1, load_best=True):

        test_loader = self._get_data('test')[1]

        # -------------------------------------------------------
        # LOAD BEST MODEL
        # -------------------------------------------------------
        if load_best:
            path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            print(f"Loading best model from: {path}")
            self.model.load_state_dict(torch.load(path, map_location=self.device))

        self.model.eval()

        preds, trues, inputs = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):

                if i >= num_batches:
                    break

                # ------------------------------------------------
                # Shapes:
                # batch_x: [B, seq_len, 321]
                # batch_y: [B, label_len + pred_len, 321]
                # ------------------------------------------------

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ------------------------------------------------
                # FORWARD PASS
                # outputs: [B, pred_len, 321]
                # ------------------------------------------------
                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                # ------------------------------------------------
                # SAVE:
                # 1) INPUT (context window)
                # 2) TRUE future (ground truth horizon)
                # 3) PREDICTED future
                # ------------------------------------------------

                inputs.append(batch_x.cpu().numpy())   # [B, seq_len, 321]
                preds.append(outputs.cpu().numpy())    # [B, pred_len, 321]
                trues.append(true.cpu().numpy())       # [B, pred_len, 321]

        # -------------------------------------------------------
        # CONCATENATE
        # -------------------------------------------------------
        inputs = np.concatenate(inputs, axis=0)   # [N, seq_len, 321]
        preds  = np.concatenate(preds, axis=0)    # [N, pred_len, 321]
        trues  = np.concatenate(trues, axis=0)    # [N, pred_len, 321]

        print("Saved sample shapes:")
        print("inputs:", inputs.shape)
        print("preds :", preds.shape)
        print("trues :", trues.shape)

        # -------------------------------------------------------
        # SAVE
        # -------------------------------------------------------
        save_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path, "sample_input.npy"), inputs)
        np.save(os.path.join(save_path, "sample_pred.npy"), preds)
        np.save(os.path.join(save_path, "sample_true.npy"), trues)

        print(f"Saved in: {save_path}")

        return inputs, preds, trues
