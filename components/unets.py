import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.utils.layers import ConvAltan,UpConvAltan,ASPP
from components.attention import SelfAttention
class Unet_Residual(nn.Module):
    def __init__(self,in_channels,out_channels,n_time=1280):
        super().__init__()
        self.group1=nn.GroupNorm(32,in_channels)
        self.conv1=ConvAltan(in_channels,out_channels,False,kernel_size=3,padding=1)

        self.group2=nn.GroupNorm(32,out_channels)
        self.conv2=ConvAltan(out_channels,out_channels,False,kernel_size=3,padding=1)

        self.linear_time = nn.Linear(n_time, out_channels)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = ConvAltan(in_channels, out_channels, False,kernel_size=1, padding=0)         

    def forward(self, feature, time):

        residue = feature
        
        feature = self.group1(feature)
        feature = F.silu(feature)
        feature = self.conv1(feature)
        time = F.silu(time)

        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.group2(merged)
        
        merged = F.silu(merged)
        
        merged = self.group2(merged)
        
        return merged + self.residual_layer(residue)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int=4, REQUEST: bool=False):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = ConvAltan(channels, channels,False, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(channels, n_head,REQUEST)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.Gelu=nn.GELU()
    def forward(self, x):

        residue_long = x
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        n, c, h, w = x.shape

        x = x.view((n, c, h * w))
        

        x = x.transpose(-1, -2)
        residue_short = x
        

        x = self.layernorm_1(x)
        
  
        x = self.attention_1(x)
        
        x += residue_short
 
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
   
        x = x * self.Gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        return self.conv_output(x) + residue_long
    
class UpConv(nn.Module):
    def __init__(self, channels,which_one):
        super().__init__()
        if(which_one==0):
            self.conv = UpConvAltan(channels,channels,False,kernel_size=4,stride=2,padding=1)
        else:
            self.conv = UpConvAltan(channels,channels,False,kernel_size=2,stride=2,padding=0)
    
    def forward(self, x):
        
        return self.conv(x)
    
class Unet_Aspp(nn.Module):
    def __init__(self,in_channels,out_channels,n_time=1280):
        super().__init__()
        self.group1=nn.GroupNorm(32,in_channels)
        self.conv1=ASPP(in_channels,out_channels)

        self.group2=nn.GroupNorm(32,out_channels)
        self.conv2=ASPP(out_channels,out_channels)

        self.linear_time = nn.Linear(n_time, out_channels)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = ConvAltan(in_channels, out_channels, False,kernel_size=1, padding=0)         

    def forward(self, feature, time):

        residue = feature
        
        feature = self.group1(feature)
        feature = F.silu(feature)
        feature = self.conv1(feature)
        time = F.silu(time)

        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.group2(merged)
        
        merged = F.silu(merged)
        
        merged = self.group2(merged)
        
        return merged + self.residual_layer(residue)
    