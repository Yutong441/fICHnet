# from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
import torch
from torch import nn
from functools import partial
from . import building_block as BB
from . import CNN_basic as CB


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        # dynamic add padding based on the kernel_size


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu',
                 *args, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = \
            in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1,
            *args, **kwargs):
        conv = partial(Conv2dAuto, kernel_size=3, bias=False)
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), 
            nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
    
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1,
            *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs,
                downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    Args:
        `in_channels`: number of channels in the input image
        `block_sizes`: number of channels in each layer
        `depths`: number of resnet blocks
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], 
            depths=[2,2,2,2], activation='relu', block=ResNetBasicBlock,
            *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        # first layer
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7,
                stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0],
                activation=activation, block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]       
        ])
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks: x = block(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, add_sigmoid='None',
            times_max=1, extra_features=0, dropout=0, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = CB.CNN2d_decoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        self.add_sigmoid = add_sigmoid
        self.times_max = times_max
        
    def forward(self, x):
        if type (x) == torch.Tensor:
            x = self.encoder(x)
            x = self.decoder(x)
        elif type (x) == list:
            encoder_x = self.encoder (x[0])
            x = self.decoder ([encoder_x, x[1]])
        return BB.act_func (x, self.add_sigmoid, self.times_max)

def resnet2d_n (in_channels, n_classes, model_type='resnet18', **kwargs):
    if model_type == 'resnet2d_18':
        return ResNet(in_channels, n_classes, block=ResNetBasicBlock,
                depths=[2, 2, 2, 2], **kwargs)
    elif model_type == 'resnet2d_34':
        return ResNet(in_channels, n_classes, block=ResNetBasicBlock,
                depths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet2d_50':
        return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock,
                depths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet2d_101':
        return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock,
                depths=[3, 4, 23, 3], **kwargs)
