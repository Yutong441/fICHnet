# decoder class that can integrate with non-imaging features to construct
# fusion net
import torch
from torch import nn


class CNN_decoder (nn.Module):
    def __init__(self, embed_chan, return_embed=False):
        '''
        Args:
            `embed_chan`: number of channels at the output of the embedding
            layer, as defined by the `embedding` function
            `return_embed`: whether to return the embedding, this
            option will be enabled automatically for a fusion model
        '''
        super().__init__()
        self.return_embed = return_embed
        self.linear = nn.Identity ()
        self.embed_chan = embed_chan
    def embedding (self, x):
        return x
    def forward (self, x):
        out= self.embedding (x)
        if self.return_embed: return out
        else: return self.linear (out)


class lin_decoder(CNN_decoder):
    def __init__(self, embed_chan):
        super().__init__(embed_chan=embed_chan)
        self.blocks = nn.Sequential(
                nn.Linear(embed_chan, embed_chan),
                nn.ReLU())

    def embedding(self, x):
        return self.blocks(x)


class fuse_decoder(nn.Module):
    def __init__(self, n_classes, decoder1, decoder2):
        '''
        Args:
            `n_classes`: the number of output class
        '''
        super().__init__()
        #self.decoder_list = decoder_list
        #chan = sum ([decode.embed_chan for decode in decoder_list])
        #for i in range(len(decoder_list)):
        #    self.decoder_list[i].return_embed = True
        self.decoder1, self.decoder2 = decoder1, decoder2
        chan = decoder1.embed_chan + decoder2.embed_chan
        self.decoder1.return_embed=True
        self.decoder2.return_embed=True
        self.linear = nn.Linear (chan, n_classes)
    def forward (self, x):
        '''
        `x` should be a list, matching the number and order of the
        `decoder_list` 
        '''
        #out = [decode (inp) for decode, inp in zip (self.decoder_list, x)]
        #out = torch.cat (out, axis=1)
        out1 = self.decoder1 (x[0])
        out2 = self.decoder2 (x[1])
        out = torch.cat ([out1, out2], axis=1)
        return self.linear (out)
