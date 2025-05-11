import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.utils.layers import ConvAltan,UpConvAltan,ASPP
from components.attention import SelfAttention
from components.unets import Unet_Residual,UNET_AttentionBlock,UpConv,Unet_Aspp

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)
        x=x.to("cuda")
        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x

class UNet(nn.Module):
    def __init__(self, c_in=4, c_out=4, time_dim=1280, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.time_emb=TimeEmbedding(320).to(device)
        self.group=nn.GroupNorm(32,320)
        self.down=nn.ModuleList([
            ConvAltan(c_in,320,False,kernel_size=3,padding=1),
            nn.ModuleList([Unet_Residual(320,320),UNET_AttentionBlock(40,8)]),
            nn.ModuleList([Unet_Residual(320,320),UNET_AttentionBlock(40,8)]),
            ConvAltan(320,320,False,kernel_size=3,stride=2,padding=1),
            nn.ModuleList([Unet_Aspp(320,640),UNET_AttentionBlock(80,8)]),
            nn.ModuleList([Unet_Residual(640,640),UNET_AttentionBlock(80,8)]),
            ConvAltan(640,640,False,kernel_size=3,stride=2,padding=1),
            nn.ModuleList([Unet_Aspp(640,1280),UNET_AttentionBlock(160,8)]),
            nn.ModuleList([Unet_Residual(1280,1280),UNET_AttentionBlock(160,8)]),
            ConvAltan(1280,1280,False,kernel_size=3,stride=2,padding=1),
            nn.ModuleList([Unet_Aspp(1280,1280)]),
            nn.ModuleList([Unet_Residual(1280,1280)])
        ])
        
        self.bottle_neck=nn.ModuleList([
            Unet_Aspp(1280,1280),
            UNET_AttentionBlock(160,8),
            Unet_Aspp(1280,1280)
        ])

        self.up=nn.ModuleList([
            nn.ModuleList([Unet_Aspp(2560,1280)]),
            nn.ModuleList([Unet_Residual(2560,1280)]),
            nn.ModuleList([Unet_Aspp(2560,1280),UpConv(1280,0)]),
            nn.ModuleList([Unet_Residual(2560,1280),UNET_AttentionBlock(160,8)]),
            nn.ModuleList([Unet_Residual(2560,1280),UNET_AttentionBlock(160,8)]),
            nn.ModuleList([Unet_Residual(1920,1280),UNET_AttentionBlock(160,8),UpConv(1280,0)]),
            nn.ModuleList([Unet_Residual(1920,640),UNET_AttentionBlock(80,8)]),
            nn.ModuleList([Unet_Residual(1280,640),UNET_AttentionBlock(80,8)]),
            nn.ModuleList([Unet_Residual(960,640),UNET_AttentionBlock(80,8),UpConv(640,0)]),
            nn.ModuleList([Unet_Aspp(960,320),UNET_AttentionBlock(40,8)]),
            nn.ModuleList([Unet_Residual(640,320),UNET_AttentionBlock(40,8)]),
            nn.ModuleList([Unet_Residual(640,320),UNET_AttentionBlock(40,8)]),
        ])

        self.outc=ConvAltan(320,c_out,False,kernel_size=3,stride=1,padding=1)
        
    def pos_encoding(self, t):
        # Shape: (160,)
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
        # Shape: (1, 160)
        x = torch.tensor([t], dtype=torch.float32)[:, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t)
        t=self.time_emb(t)
        # DOWN PART
        skip_connections = []
        for sub_module_list in self.down:
            if(isinstance(sub_module_list,nn.ModuleList)):
                for layers in sub_module_list:
                    if(isinstance(layers,Unet_Residual) or isinstance(layers,Unet_Aspp)):
                        x=layers(x,t)
                    else:
                        x=layers(x)
                skip_connections.append(x)
            else:
                x=sub_module_list(x)
                skip_connections.append(x)

            

        #bottleneck
        for layers in self.bottle_neck:
            if(isinstance(layers,Unet_Aspp)):
                x=layers(x,t)
            else:
                x=layers(x)
            
        #UpBone
        for sub_module_lists in self.up:
            x=torch.cat((x,skip_connections.pop()),dim=1)
            for layers in sub_module_lists:
                if(isinstance(layers,Unet_Residual) or isinstance(layers,Unet_Aspp)):
                    
                    x=layers(x,t)
                else:
                    x=layers(x)
        x=self.group(x)
        x=F.silu(x)
        x = self.outc(x)
        return x
