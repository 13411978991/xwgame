import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
import torch

from itertools import repeat
import copy


#掩盖模块，按照mask_rate概率随机掩盖输入连续row*mask_rate行数据为0
class mask_block(nn.Module):
    def __init__(self, mask_rate):
        super(mask_block, self).__init__()
        self.mask_rate = mask_rate

    def forward(self, input):
        if not self.training:
            return input
        assert input.dim() == 4
        batch_size = input.shape[0]
        len_mask = round(input.shape[2] * self.mask_rate)
        b=torch.randint(low=0,high=input.shape[2]-len_mask,size=[batch_size])
        for i in range(batch_size):
          input[i,:,b[i]:b[i]+len_mask,:]=0
        return input


