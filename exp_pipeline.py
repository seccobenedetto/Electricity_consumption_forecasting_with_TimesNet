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
        if self.args.use_gpu:
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
        """
        Model expects ONLY:
            forward(x_enc, x_dec)

        where:
        x_enc: [B, seq_len, 321]
        x_dec: [B, label_len + pred_len, 321]
        """

        model_dict = { 'TimesNet': TimesNet, }
        model = model_dict[self.args.model](self.args).float()

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
    # VALIDATION
    # -------------------------------------------------------
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

                # decoder input:
                # first part = known labels 
                # last part = zeros for future 
                #
                # shape: [B, label_len + pred_len, 321]
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]) # zeros for future predictions
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1 
                    ).to(self.device) # here we concatenate the known labels with the zeros for future predictions

                # forward WITHOUT time embeddings
                # outputs: [B, pred_len, 321]
                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                loss = criterion(outputs, true)
                total_loss.append(loss.item())

        self.model.train()

        return np.mean(total_loss)

    # -------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------
    def train(self, setting):

        train_loader = self._get_data('train')[1]
        vali_loader = self._get_data('val')[1]
        test_loader = self._get_data('test')[1]

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        early_stopping = EarlyStopping(patience=self.args.patience)

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            epoch_time = time.time()
            iter_count = 0

            for i, (batch_x, batch_y) in enumerate(train_loader):

                iter_count += 1
                # ------------------------------------------------
                # batch_x: [B, seq_len, 321]
                # batch_y: [B, label_len + pred_len, 321]
                # ------------------------------------------------

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                optimizer.zero_grad()

                # decoder input (NO time encoding)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                loss = criterion(outputs, true)
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"Epoch {epoch} | Iter {i} | Loss {loss.item():.6f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"Speed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(
                f"[Epoch {epoch}] "
                f"Time cost: {time.time() - epoch_time:.2f}s, "
                f"train={np.mean(train_losses):.6f} "
                f"vali={vali_loss:.6f} "
                f"test={test_loss:.6f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        # here we load the best model saved during training (the one with the lowest validation loss)
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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                preds.append(outputs.cpu().numpy())
                trues.append(true.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print("Test shapes:", preds.shape, trues.shape)

        mse, mae = metric(preds, trues)

        print(f"MSE={mse:.6f}, MAE={mae:.6f}")

        return mse, mae