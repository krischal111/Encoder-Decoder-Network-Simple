import torch.nn.functional as F

from mynet_parts import *

class myNet(nn.Module):
    def __init__(self):

        super(myNet, self).__init__()
        self.n_channels = 1

        self.inc = DoubleConv(self.n_channels, 2) # => same dimensions 2 channels
        self.down1 = Down(2, 3)
        self.down2 = Down(3, 4)
        self.up1 = Up(4, 3)
        self.up2 = Up(3, 2) # => same dimensions 2 channels
        self.outc = OutConv(2, 1)
    
    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.outc(x)
