# spatial transformer network 
# modified from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
import torch
from torch import nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self, in_chan, img_shape):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(in_chan, 8, kernel_size=(3,7,7), padding=(1,3,3)),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 10, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(10, 10, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        D, H, W= [((i//2)//2)//2 for i in img_shape]
        self.num_pixel= D*H*W*10
        self.fc_loc = nn.Sequential(
            nn.Linear(self.num_pixel, 32),
            nn.ReLU(True),
            nn.Linear(32, 4 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()

    def forward (self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.num_pixel)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4) #3x4 for 3D data

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
