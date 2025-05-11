import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.utils.layers import ConvAltan,UpConvAltan
from components.vaes import ResidualBlock,ResidualBlock4Aspp,Attention4VAE

class Vaedecoder(nn.Sequential):
    """Some Information about Vaedecoder"""
    def __init__(self):
        super(Vaedecoder, self).__init__(
            ConvAltan(4,4,False,kernel_size=1,stride=1,padding=0),
            ConvAltan(4,512,False,kernel_size=3,stride=1,padding=1),
            ResidualBlock4Aspp(512,512),
            Attention4VAE(512), 

            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock4Aspp(512, 512), 
            ResidualBlock(512, 512), 

            UpConvAltan(512,512,kernel_size=4,stride=2,padding=1),
            ConvAltan(512, 512,False, kernel_size=3, padding=1), 
            ResidualBlock4Aspp(512,512), 
            ResidualBlock(512,512),
            ResidualBlock(512,512),

            UpConvAltan(512,512,kernel_size=4,stride=2,padding=1),
            ConvAltan(512, 512,False, kernel_size=3, padding=1), 
            ResidualBlock4Aspp(512,256), 
            ResidualBlock(256,256),
            ResidualBlock(256,256),

            UpConvAltan(256,256,kernel_size=4,stride=2,padding=1),
            ConvAltan(256,256,False, kernel_size=3, padding=1), 
            ResidualBlock4Aspp(256,128), 
            ResidualBlock(128,128),
            ResidualBlock(128,128),
            
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            ConvAltan(128, 3,False, kernel_size=3, padding=1), 
            
        )

    def forward(self, x):
        x /= 0.18215
        for method in self:
            x=method(x)
        return x