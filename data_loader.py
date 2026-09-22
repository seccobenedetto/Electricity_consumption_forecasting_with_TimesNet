import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# the Dataset class is used in general to create custom datasets for PyTorch, and DataLoader is used to load data in batches.
from sklearn.preprocessing import StandardScaler


# ============================================================
# DATA PROVIDER
# ============================================================
# This function is used to create a dataset and a dataloader for the electricity dataset. 
# It takes in the arguments and a flag (train/val/test) and returns the dataset and dataloader.
def data_provider(args, flag):
    """
    args must contain:
        - root_path
        - data_path (electricity.csv)
        - seq_len
        - label_len
        - pred_len
        - batch_size
        - num_workers
    """
    # # If the flag is 'train', we shuffle the data, otherwise we don't. 
    # We shuffle the starting indices of windows, not the time series itself.
    shuffle_flag = (flag == 'train') 
    drop_last = False # here we don't drop the last batch

    dataset = ElectricityDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        flag=flag,
        scale=True
    )

    print(f"{flag} samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return dataset, loader


# ============================================================
# ELECTRICITY DATASET
# ============================================================
# Here we define a custom dataset class for the electricity forecasting task. 
# This class inherits from PyTorch's Dataset class and implements the necessary methods to load and preprocess the data.
class ElectricityDataset(Dataset):
    """
    Data shape (raw CSV):
        df_raw: [T=26304, C=322]
            - column 0: date
            - columns 1..321: time series variables

    After processing:
        data_x: [T, C]
        data_y: [T, C]
    """

    def __init__(
        self,
        root_path,
        data_path,
        seq_len=96,   # The length of the input sequence (number of time steps) used for prediction.
        label_len=48, # The length of the label sequence (number of time steps) used for prediction.
        pred_len=96,  # The length of the prediction sequence (number of time steps) to predict.
        flag='train',
        scale=True
    ):
        super().__init__()

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.flag = flag
        self.scale = scale

        # ------------------------------------------------------------
        # 1. LOAD CSV
        # ------------------------------------------------------------
        file_path = os.path.join(root_path, data_path)
        df_raw = pd.read_csv(file_path)

        # EXPECTED:
        # df_raw.shape = (26304, 322)
        # columns: ["date", f1, f2, ..., f321]

        # ------------------------------------------------------------
        # 2. REMOVE DATE COLUMN
        # ------------------------------------------------------------
        if "date" in df_raw.columns:
            df_data = df_raw.drop(columns=["date"])
        else:
            df_data = df_raw

        # df_data shape: [26304, 321]
        # PyTorch models expect input data to be float32
        data = df_data.values.astype(np.float32)

        # ------------------------------------------------------------
        # 3. TRAIN/VAL/TEST SPLIT (standard TimesNet-style)
        # ------------------------------------------------------------
        num_train = int(len(data) * 0.7)
        num_val = int(len(data) * 0.1)
        num_test = len(data) - num_train - num_val

        """
        Input  x: [s, s + seq_len)
        Target y: [s + seq_len - label_len, s + seq_len + pred_len), so the target overlaps with input by label_len.
        
                |s                              |s+seq_len      |s+seq_len+pred_len
        ---------------------------------------------------------------> time arrow 

                [=========== seq_len ===========]                    INPUT WINDOW 
                            [==== label_len ====]                    overlap
                                                 [== pred_len ==]    FUTURE
        """

        # here we define the borders for the train, validation, and test sets.
        # border1s defines the starting index for each set, and border2s defines the ending index for each set.
        # we shift by seq_len to ensure that the input sequence is fully contained within the dataset.
        border1s = [ 0,            num_train - seq_len,  num_train + num_val - seq_len ]
        border2s = [ num_train,    num_train + num_val,  len(data)]

        if flag == 'train':         border1, border2 = border1s[0], border2s[0]
        elif flag == 'val':         border1, border2 = border1s[1], border2s[1]
        else:                       border1, border2 = border1s[2], border2s[2]

        # ------------------------------------------------------------
        # 4. NORMALIZATION (fit ONLY on train cause we don't want to leak information from the validation/test sets)
        # ------------------------------------------------------------
        self.scaler = StandardScaler()

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        # ------------------------------------------------------------
        # 5. STORE SPLIT DATA
        # ------------------------------------------------------------
        self.data_x = data[border1:border2]  # input series. shape: (N_train, 321)
        self.data_y = data[border1:border2]  # target (same space). shape: (N_train, 321)

    # ============================================================
    # 6. SLIDING WINDOW SAMPLING
    # ============================================================
    # This function gives me the index-th training sample, given by:
    # a. the sliding window of length seq_len from the input series, 
    # b. the corresponding target sequence of length label_len + pred_len
    def __getitem__(self, index):
        """
        Returns:
            x: [seq_len, C]
            y: [label_len + pred_len, C]
        """
        # input sequence: [s, s + seq_len)
        s_begin = index
        s_end = s_begin + self.seq_len
        # target sequence: [s + seq_len - label_len, s + seq_len + pred_len)
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # [seq_len, C]
        seq_y = self.data_y[r_begin:r_end]  # [label_len+pred_len, C]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    # ------------------------------------------------------------
    # inverse normalization (for evaluation)
    # ------------------------------------------------------------
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)