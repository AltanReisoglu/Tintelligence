import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.attention import SelfAttention
from components.utils.layers import ConvAltan,ASPP
class Attention4VAE(nn.Module):
    def __init__(self, channels,n_head=1,request=False):
        super().__init__()
        self.b_norm=nn.BatchNorm2d(channels)
        self.attention=SelfAttention(channels,n_head,request)
        
    def forward(self,x):
        residue=x
        B,C,H,W=x.shape
        x=x.view((B,C,H*W))
        x=x.transpose(-1,-2)
        x=self.attention(x)
        x=x.transpose(-1,-2)
        x=x.view((B,C,H,W))
        x+=residue
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 =ConvAltan(in_channels, out_channels,False, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = ConvAltan(out_channels, out_channels,False, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return x + self.residual_layer(residue)
    
class ResidualBlock4Aspp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 =ASPP(in_channels, out_channels)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = ASPP(out_channels, out_channels)
        self.gelu=nn.GELU(approximate="tanh")

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = self.gelu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = self.gelu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)