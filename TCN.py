import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ============================================================
# 1. REMOVE RIGHT PADDING (causality enforcement)
# ============================================================
class Chomp1d(nn.Module):
    """
    Removes the extra right padding introduced by dilated convolutions to ensure causality (no future leakage).
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: [B, C, T + padding] → remove last padding elements
        return x[:, :, :-self.chomp_size].contiguous()


# ============================================================
# 2. TEMPORAL BLOCK (2-layer dilated causal conv + residual)
# ============================================================
class TemporalBlock(nn.Module):
    """
    A single TCN residual block:
    Conv1D(dilated) → ReLU → Dropout → Conv1D(dilated) → ReLU → Dropout + residual
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation  # ensures same length after chomp

        # ---------------- FIRST CONV ----------------
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # ---------------- SECOND CONV ----------------
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # ---------------- RESIDUAL ----------------
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T]

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # residual connection
        res = x if self.downsample is None else self.downsample(x)

        return self.final_relu(out + res)


# ============================================================
# 3. FULL TCN BACKBONE
# ============================================================
class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation: 1, 2, 4, 8, ...
    """

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):

            dilation_size = 2 ** i  # exponential receptive field

            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T]
        return self.network(x)


# ============================================================
# 4. FINAL MODEL (TimesNet-style wrapper)
# ============================================================
class Model(nn.Module):
    """
    TCN baseline model for fair comparison with TimesNet.

    Input : [B, seq_len, enc_in]
    Output: [B, pred_len, c_out]
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ---------------- INPUT PROJECTION ----------------
        # same idea as TimesNet: map 321 → d_model
        self.input_projection = nn.Linear(configs.enc_in, configs.d_model)

        # ---------------- TCN BACKBONE ----------------
        # channels define feature size evolution
        channels = [configs.d_model] * configs.e_layers

        self.tcn = TemporalConvNet(
            num_inputs=configs.d_model,
            num_channels=channels,
            kernel_size=3,
            dropout=0.1
        )

        # ---------------- OUTPUT PROJECTION ----------------
        self.output_projection = nn.Linear(configs.d_model, configs.c_out)

        # ---------------- TEMPORAL HEAD ----------------
        # compress T → pred_len (like TimesNet predict_linear but simpler)
        self.temporal_head = nn.Linear(self.seq_len, self.pred_len)

    # ========================================================
    def forward(self, x):
        """
        x: [B, T, C]
        """

        B, T, C = x.shape

        # ====================================================
        # 1. NORMALIZATION (same spirit as TimesNet)
        # ====================================================
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means

        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # ====================================================
        # 2. INPUT EMBEDDING
        # ====================================================
        x = self.input_projection(x)  # [B, T, d_model]

        # ====================================================
        # 3. TCN EXPECTS [B, C, T]
        # ====================================================
        x = x.permute(0, 2, 1)  # [B, d_model, T]

        # ====================================================
        # 4. TEMPORAL FEATURE EXTRACTION
        # ====================================================
        x = self.tcn(x)  # [B, d_model, T]

        # ====================================================
        # 5. BACK TO [B, T, d_model]
        # ====================================================
        x = x.permute(0, 2, 1)

        # ====================================================
        # 6. TEMPORAL COMPRESSION (T → pred_len)
        # ====================================================
        x = self.temporal_head(x.permute(0, 2, 1)).permute(0, 2, 1)

        # ====================================================
        # 7. OUTPUT PROJECTION
        # ====================================================
        x = self.output_projection(x)  # [B, pred_len, C]

        # ====================================================
        # 8. DE-NORMALIZATION
        # ====================================================
        x = x * (stdev[:, 0, :].unsqueeze(1))
        x = x + (means[:, 0, :].unsqueeze(1))

        return x