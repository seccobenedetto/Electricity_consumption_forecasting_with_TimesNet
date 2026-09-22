import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
# from Embed import DataEmbedding
from Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    This function computes the Fast Fourier Transform (FFT) of the input tensor x 
    and identifies the top k significant periods based on the amplitude of the frequency components.

    Input:
    - x: A tensor of shape [B, T, C], where B is the batch size, T is the length of the time series, and C is the number of features/channels.
    - k: The number of top significant periods to identify (this is a hyperparameter that can be tuned)

    Output:
    - period: A list of shape [k], containing the top k significant periods.
    - period_weight: A tensor of shape [B, k], containing the corresponding amplitudes of the top k frequencies for each sample in the batch.
    """
    
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    # rfft = real FFT, so only positive frequencies
    xf = torch.fft.rfft(x, dim=1) # output: [B, T_freq, N]

    # find period by amplitudes: here we assume that the periodic features are basically constant
    # in different batch and channel, so we mean out these two dimensions, getting a list frequency_list with shape[T] 
    # each element at pos t of frequency_list denotes the overall amplitude at frequency (t)
    frequency_list = abs(xf).mean(0).mean(-1) 
    frequency_list[0] = 0 # frequency 0 = constant component (mean of signal), not useful for periodicity

    #by torch.topk(),we can get the biggest k elements of frequency_list, and its positions(i.e. the k-main frequencies in top_list)
    _, top_list = torch.topk(frequency_list, k)

    #Returns a new Tensor 'top_list', detached from the current graph.
    #The result will never require gradient.Convert to a numpy instance
    top_list = top_list.detach().cpu().numpy()
     
    #period:a list of shape [top_k], recording the periods of mean frequencies respectively
    period = x.shape[1] // top_list
    # In reality p = T / f (not necessarily integer), but we force it: this is a design compromise.

    # Here,the 2nd item returned has a shape of [B, top_k],representing the biggest top_k amplitudes 
    # for each piece of data, with N features being averaged.
    # period_weight[b, i] = amplitude of i-th frequency for sample b
    return period, abs(xf).mean(-1)[:, top_list] # returns: period (shape [k]) and weights (shape [B,k])
    


class TimesBlock(nn.Module):
    """
    TimesBlock is the core component of TimesNet, which captures temporal patterns in time series data 
    using a combination of Fast Fourier Transform (FFT) and convolutional operations.
    """

    def __init__(self, configs):    # configs is the configuration defined for TimesBlock
        super(TimesBlock, self).__init__() 
        self.seq_len = configs.seq_len   # sequence length 
        self.pred_len = configs.pred_len # prediction length
        self.k = configs.top_k           # k denotes how many top frequencies are taken into consideration

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, # d_model is the embedding dimension, d_ff is the hidden dimension of the feedforward network
                            num_kernels=configs.num_kernels), # num_kernels is the number of convolutional kernels in the Inception block
            nn.GELU(),                                        # GELU activation function, which is a smooth approximation of ReLU
            Inception_Block_V1(configs.d_ff, configs.d_model,
                            num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
            #B: batch size  T: length of time series  N:number of features/channels

        # ======================== FFT ========================
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = [] # to store the results of the top k periods after convolution and reshaping

        for i in range(self.k):         # cycle over the number of signifcant chosen periods
            period = period_list[i]

            # ====================== ADD PADDING ========================
            # padding: to form a 2D map, we need total length of the sequence, plus the part 
            # to be predicted, to be divisible by the period, so padding is needed
            if (self.seq_len + self.pred_len) % period != 0:
                length = ( ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # ======================= RESHAPING to 2D ========================
            # reshape: we need each channel of a single piece of data to be a 2D variable,
            # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions 
            # to be convolutioned to the last 2 dimensions, by calling the permute() func.
            # Whereafter, to make the tensor contiguous in memory, call contiguous()
            out = out.reshape(B, length // period, period,  N).permute(0, 3, 1, 2).contiguous()

            # ======================= 2D CONVOLUTION =======================
            # 2D convolution to grasp the intra- and inter- period information
            out = self.conv(out)  # in our case, it's a forward pass inside the Inception blocks

            # ======================= RESHAPING to 1D ========================
            # reshape back, similar to reshape to 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)  # now [B, length, N]

            # ====================== REMOVE PADDING ========================
            # truncating down the padded part of the output and put it to result
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # stack all results
        res = torch.stack(res, dim=-1) #res: 4D [B, length , N, top_k]

        # adaptive aggregation
        # First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        period_weight = F.softmax(period_weight, dim=1) 

        # after two unsqueeze(1), shape -> [B,1,1,top_k], so repeat the weight to fit the shape of res
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        
        #add by weight the top_k periods' result, getting the result of this TimesBlock
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
    
        return res
    


class Model(nn.Module):
    """
    Simplified TimesNet model for long-term forecasting on electricity dataset.
    No time encoding, no multi-task logic.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # basic params
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # stack of TimesBlocks
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        self.layer_norm = nn.LayerNorm(configs.d_model)

        # input embedding: simple linear projection (no time features)
        self.input_projection = nn.Linear(configs.enc_in, configs.d_model)

        # temporal projection: expand T → T + pred_len
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)

        # output projection: back to original dimension
        self.output_projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x):
        """
        x: [B, seq_len, C]
        """

        # ================= NORMALIZATION =================
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means

        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # ================= EMBEDDING =================
        # [B, T, C] → [B, T, d_model]
        x = self.input_projection(x)

        # ================= TEMPORAL EXPANSION =================
        # operate on time dimension
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        # now [B, T + pred_len, d_model]

        # ================= TIMESBLOCK STACK =================
        for block in self.model:
            x = self.layer_norm(block(x))

        # ================= PROJECTION =================
        x = self.output_projection(x)
        # [B, T + pred_len, C]

        # ================= DE-NORMALIZATION =================
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1))

        # ================= RETURN ONLY PREDICTION =================
        return x[:, -self.pred_len:, :]
