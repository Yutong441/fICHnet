#from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/densenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from . import building_block as BB
from . import CNN_basic as CB


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv2d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv2d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
        no_max_pool=False, growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, dropout=0, num_classes=1000,
        add_sigmoid='None', times_max=1, *args, **kwargs):

        super().__init__()

        # First convolution
        self.encoder = [('conv1',
                          nn.Conv2d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(7, 7),
                                    stride=(2, 2),
                                    padding=(3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm2d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.encoder.append(
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
        self.encoder = nn.Sequential(OrderedDict(self.encoder))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=dropout)
            self.encoder.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.encoder.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.encoder.add_module('norm5', nn.BatchNorm2d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.add_sigmoid = add_sigmoid
        self.times_max = times_max
        self.num_features = num_features
        self.decoder = CB.CNN2d_decoder(num_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = F.relu(features, inplace=True)
        out = self.decoder(out)
        return BB.act_func(out, self.add_sigmoid, self.times_max)


def densenet2d_n (in_channels, n_classes, model_type, **kwargs):
    if model_type == 'densenet2d_121':
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         n_input_channels=in_channels, 
                         num_classes=n_classes,
                         **kwargs)
    elif model_type == 'densenet2d_169':
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         n_input_channels=in_channels, 
                         num_classes=n_classes,
                         **kwargs)
    elif model_type == 'densenet2d_201':
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         n_input_channels=in_channels, 
                         num_classes=n_classes,
                         **kwargs)
    elif model_type == 'densenet2d_264':
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         n_input_channels=in_channels, 
                         num_classes=n_classes,
                         **kwargs)
    return model
