# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiFunctional import _wrap_tensor
import torch.nn as nn
import torch

class AverageBackProjection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(AverageBackProjection,self).__init__()
        # in_channels is the number of channels before the average calculation
        # out_channels is the number of channels after average calculation
        # out_channels < in_channels
        self.times = in_channels//out_channels
        self.out_channels = out_channels
        self.cs0 = ME.MinkowskiConvolution(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,dimension=3)
        self.cs1 = ME.MinkowskiConvolution(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,dimension=3)
        self.cs2 = ME.MinkowskiConvolution(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,dimension=3)

    def forward(self,x):

        #Calculate the average in the middle
        n, c = x.size()
        downsampled = torch.mean(x.F.view(n, c//self.out_channels, self.out_channels), dim=1)
        
        upsampled = downsampled.repeat(1, self.times)
        upsampled = _wrap_tensor(x,upsampled)
        
        residual = x-upsampled
        residual = self.cs0(residual)
        residual = self.cs1(residual)
        residual = self.cs2(residual)

        downsampled = _wrap_tensor(x,downsampled)

        return downsampled+residual


class CopyBackProjection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CopyBackProjection,self).__init__()
        # in_channels is the number of channels before the copy calculation
        # out_channels is the number of channels after copy calculation
        # out_channels > in_channels

        self.times = out_channels//in_channels
        self.out_channels = out_channels

        self.csd0 = ME.MinkowskiConvolution(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,dimension=3)
        self.csd1 = ME.MinkowskiConvolution(in_channels=out_channels,out_channels=in_channels,kernel_size=kernel_size,dimension=3)
        self.csd2 = ME.MinkowskiConvolution(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,dimension=3)

        self.csu0 = ME.MinkowskiConvolution(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,dimension=3)
        self.csu1 = ME.MinkowskiConvolution(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,dimension=3)
        self.csu2 = ME.MinkowskiConvolution(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,dimension=3)

    def forward(self,x):

        upsampled = x.F.repeat(1, self.times)
        upsampled = upsampled = _wrap_tensor(x,upsampled)

        downsampled = self.csd0(upsampled)
        downsampled = self.csd1(downsampled)
        downsampled = self.csd2(downsampled)

        residual = x - downsampled
        residual = self.csu0(residual)
        residual = self.csu1(residual)
        residual = self.csu2(residual)        

        return upsampled+residual



class BackProjectionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BackProjectionModule,self).__init__()
        self.ABP = AverageBackProjection(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)
        self.CBP = CopyBackProjection(in_channels=out_channels,out_channels=in_channels,kernel_size=kernel_size)

    def forward(self,x,reverse=False):

        if not reverse:
            x = self.ABP(x)
        else:
            x = self.CBP(x)

        return x