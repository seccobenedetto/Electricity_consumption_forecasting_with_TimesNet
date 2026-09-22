# in this file we define the inception blocks used in the Timeblock model

import torch
import torch.nn as nn


# this block is used to extract features from the input image
# it uses multiple convolutional kernels with different sizes to capture different features
# the peculiarity of v1 is that it uses square kernels of size 1, 3, 5, 7, 9, 11
# the idea is to capture features at different scales and combine them to get a richer representation of the input image
class Inception_Block_V1(nn.Module):

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels    
        self.out_channels = out_channels
        self.num_kernels = num_kernels   
        kernels = []

        # create convolutional kernels with different sizes: 1, 3, 5, 7, 9, 11
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
            # padding is needed to keep the output size the same as the input size

        self.kernels = nn.ModuleList(kernels) # store the kernels in a ModuleList so that they are registered as submodules

        if init_weight:
            self._initialize_weights()

    # initialize the weights of the convolutional kernels using kaiming normal initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        # apply each kernel to the input and store the results in a list
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        # stack the results along a new dimension and take the mean along that dimension
        res = torch.stack(res_list, dim=-1).mean(-1)
        # output has shape (batch_size, out_channels, height, width)
        return res





# this block is used to extract features from the input image
# the peculiarity of v2 is that it uses rectangular kernels of size 1x3, 3x1, 1x5, 5x1, 1x7, 7x1, ..., 1x(2n+1), (2n+1)x1
# so the idea is to capture features in both horizontal and vertical directions
class Inception_Block_V2(nn.Module):

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        
        # create convolutional kernels with different sizes: 1x3, 3x1, 1x5, 5x1, 1x7, 7x1, ..., 1x(2n+1), (2n+1)x1
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        # add a 1x1 convolutional kernel to the list of kernels
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    # initialize the weights of the convolutional kernels using kaiming normal initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
