import torch
from torch import nn
from .SElayer import SELayer, ECA


def act_func (x, func, multiply=1):
    if func=='sigmoid': return torch.sigmoid (x)*multiply
    elif func=='relu': return torch.relu (x)*multiply
    elif func=='tanh': return torch.tanh (x)*multiply
    else: return x

def multi_linear (in_chan, layer, step):
    '''
    Stack multiple linear layers with channels decreasing by increments
    determined by `step`
    '''
    if layer == 0: return nn.Identity ()
    else:
        block = []
        for i in range (layer):
            channels = in_chan//(step**i)
            block.extend ([nn.Linear (channels, channels//step), nn.ReLU () ])
        return nn.Sequential (*block)

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0]//2, self.kernel_size[1]//2) 

class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0]//2, 
                self.kernel_size[1]//2, self.kernel_size [2]//2) 
        # dynamic add padding based on the kernel_size

class Conv2d_in_3d (nn.Conv3d):
    def __init__(self, in_chan, out_chan, kernel=3, stride=2, *args, **kwargs):
        super().__init__(in_chan, out_chan, kernel_size= (1, kernel, kernel), 
                stride=(1, stride, stride), *args, **kwargs)
        self.padding =  (0, kernel//2, kernel//2) 
        # dynamic add padding based on the kernel_size

class Conv1d_in_3d (nn.Conv3d):
    def __init__(self, in_chan, out_chan, kernel=3, stride=1, *args, **kwargs):
        super().__init__(in_chan, out_chan, kernel_size= (kernel, 1, 1), 
                stride=1, *args, **kwargs)
        self.padding =  (kernel//2, 0, 0) 

def attention_block (attention, out_channels):
    if attention == 'se': return SELayer (out_channels)
    elif attention == 'eca': return ECA (out_channels)
