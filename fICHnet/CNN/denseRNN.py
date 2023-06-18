import torch.nn.functional as F
from . import building_block as BB
from . import decoder as DE
from .resRNN import RNN_decoder
from .densenet_2d import DenseNet


class denseCRNN(DenseNet):
    def __init__(self, in_channels, n_classes, add_sigmoid='none', times_max=1,
            step=4, extra_features=0, n_decoder=2, dropout=0, *args, **kwargs):
        super().__init__(n_input_channels=in_channels, num_classes=n_classes,
                dropout=dropout, add_sigmoid=add_sigmoid, times_max=times_max,
                *args, **kwargs)
        out_decoder = RNN_decoder (self.num_features, n_classes, step=step,
                n_decoder=n_decoder, dropout=dropout)
        if extra_features >0:
            self.decoder = DE.fuse_decoder (n_classes=n_classes,
                decoder1= out_decoder, 
                decoder2 = DE.lin_decoder (extra_features))
        else: self.decoder= out_decoder
        
    def forward(self, x):
        ''' `x`: [B, C, D, H, W] '''
        if type(x)!=tuple and type(x) != list: inp = x
        else: inp = x[0]

        B, C, D, H, W = inp.shape
        inp = inp.permute (2,0,1,3,4) # [D, B, C, H, W]
        encode_x = self.encoder(inp.reshape(-1, C, H, W) )
        _, C2, H2, W2 = encode_x.shape
        encode_x = encode_x.reshape(D, B, C2, H2, W2)
        encode_x = F.relu(encode_x, inplace=True)

        if type(x)!=tuple and type(x) != list: out = encode_x
        else: out = [encode_x, x[1]]
        out = self.decoder(out)
        return BB.act_func (out, self.add_sigmoid, self.times_max)

def denseCRNN_n (in_channels, n_classes, model_type='denseCRNN121', **kwargs):
    if model_type == 'denseCRNN121':
        model = denseCRNN ( in_channels=in_channels, 
                         n_classes=n_classes,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_type == 'denseCRNN169':
        model = denseCRNN ( in_channels=in_channels, 
                         n_classes=n_classes,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_type == 'denseCRNN201':
        model = denseCRNN ( in_channels=in_channels, 
                         n_classes=n_classes,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_type == 'denseCRNN269':
        model = denseCRNN ( in_channels=in_channels, 
                         n_classes=n_classes,
                         block_config=(6, 12, 64, 48),
                         **kwargs)
    return model
