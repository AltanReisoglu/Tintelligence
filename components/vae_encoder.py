import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.utils.layers import ConvAltan
from components.vaes import ResidualBlock,ResidualBlock4Aspp,Attention4VAE
class VAEencoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvAltan(3,128,False,kernel_size=3,stride=1,padding=1),
            ResidualBlock(128,128),
            ResidualBlock4Aspp(128,128),
            
            ConvAltan(128,128,False,kernel_size=3,stride=2,padding=1),
            ResidualBlock4Aspp(128,256),
            ResidualBlock(256,256),
            

            ConvAltan(256,256,False,kernel_size=3,stride=2,padding=1),
            ResidualBlock4Aspp(256,512),
            ResidualBlock(512,512),
            

            ConvAltan(512,512,False,kernel_size=3,stride=2,padding=1),
            ResidualBlock4Aspp(512,512),
            ResidualBlock(512,512),
            Attention4VAE(512),
            ResidualBlock(512,512),

            nn.GroupNorm(32, 512), 
            nn.SiLU(), 

            ConvAltan(512, 8,False, kernel_size=3, padding=1), 
            ConvAltan(8, 8,True, kernel_size=1, padding=0), 
            
        )
    
    def forward(self,x,noise):
        for method in self:
            if getattr(method, 'stride', None) == (2, 2):  
                x = F.pad(x, (0, 1, 0, 1))
            
            x=method(x)
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x
