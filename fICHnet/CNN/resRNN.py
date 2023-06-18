from torch import nn
from .resnet_2d import ResNetEncoder, ResNetBasicBlock, ResNetBottleNeckBlock
from . import building_block as BB
from . import decoder as DE


class RNN_decoder (DE.CNN_decoder):
    def __init__(self, in_channels, n_classes, step=4, n_decoder=2, dropout=0):
        assert n_decoder >= 2, \
                'number of decoder layers must be at least 2'
        embed_chan = in_channels//(step**(n_decoder-1))
        super().__init__(embed_chan=embed_chan)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.GRU(
                in_channels, in_channels//step, bias=True,
                batch_first=False, dropout=dropout, num_layers=2)
        self.hidden = BB.multi_linear(
                in_channels//step,
                layer=n_decoder-2, step=step)
        # minus 2 becausee the GRU and final linear layers count as the number
        # of decoder layers
        self.linear = nn.Linear(embed_chan, n_classes)

    def embedding(self, x):
        D, B, C, H, W = x.shape
        inp = x.reshape(-1, C, H, W)
        avg = self.avg(inp)  # [L*B, C, 1, 1]
        rnn_out = self.rnn(avg.reshape(D, B, C))
        out = self.hidden(rnn_out[0][-1])  # B, C_{out}
        return out


class resCRNN (nn.Module):
    def __init__(self, in_channels, n_classes, add_sigmoid='none',
                 times_max=1, step=4, deepths=[2, 2, 2, 2],
                 extra_features=0, n_decoder=2, block=ResNetBasicBlock,
                 dropout=0, *args, **kwargs):
        super().__init__()
        self.add_sigmoid = add_sigmoid
        self.times_max = times_max
        self.encoder = ResNetEncoder(in_channels, deepths=deepths, block=block)
        N_in = self.encoder.blocks[-1].blocks[-1].expanded_channels

        out_decoder = RNN_decoder(N_in, n_classes, step=step,
                                  n_decoder=n_decoder, dropout=dropout)
        if extra_features > 0:
            self.decoder = DE.fuse_decoder(
                n_classes=n_classes,
                decoder1=out_decoder,
                decoder2=DE.lin_decoder(extra_features))
        else:
            self.decoder = out_decoder

    def forward(self, x):
        ''' `x`: [B, C, D, H, W] '''
        if type(x)!=tuple and type(x) != list: inp = x
        else: inp = x[0]

        B, C, D, H, W = inp.shape
        inp = inp.permute (2,0,1,3,4) # [D, B, C, H, W]
        encode_x = self.encoder(inp.reshape(-1, C, H, W) )
        _, C2, H2, W2 = encode_x.shape
        encode_x = encode_x.reshape(D, B, C2, H2, W2)

        if type(x)!=tuple and type(x) != list: out = encode_x
        else: out = [encode_x, x[1]]
        out = self.decoder(out)
        return BB.act_func(out, self.add_sigmoid, self.times_max)


def resCRNN_n(in_channels, n_classes, model_type='resnet18', **kwargs):
    if model_type == 'resCRNN18':
        return resCRNN (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[2, 2, 2, 2], **kwargs)
    elif model_type == 'resCRNN9':
        return resCRNN (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[1, 1, 1, 1], **kwargs)
    elif model_type == 'resCRNN14':
        return resCRNN (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[1, 2, 1, 2], **kwargs)
    elif model_type == 'resCRNN34':
        return resCRNN (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resCRNN50':
        return resCRNN (in_channels, n_classes, block=ResNetBottleNeckBlock,
                deepths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resCRNN101':
        return resCRNN (in_channels, n_classes, block=ResNetBottleNeckBlock,
                deepths=[3, 4, 23, 3], **kwargs)
