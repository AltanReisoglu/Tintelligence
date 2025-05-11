
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvAltan(nn.Module):
    
    def __init__(self,in_channels,out_channels,bn=False,**kwargs):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn
    def forward(self, x):
        if(self.use_bn_act):
            x=self.conv1(x)
            x=self.bn(x)
            x=self.leaky(x)
        else:
            x=self.conv1(x)
        return x
    
class UpConvAltan(nn.Module):
    
    def __init__(self,in_channels,out_channels,bn=False,**kwargs):
        super().__init__()
        self.conv1=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn
    def forward(self, x):
        if(self.use_bn_act):
            x=self.conv1(x)
            x=self.bn(x)
            x=self.leaky(x)
        else:
            x=self.conv1(x)
        return x
    

    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = ConvAltan(in_channels, out_channels,False, kernel_size=1, padding=0, dilation=1)
        self.atrous_block6 = ConvAltan(in_channels, out_channels, False,kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = ConvAltan(in_channels, out_channels, False,kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = ConvAltan(in_channels, out_channels, False,kernel_size=3, padding=18, dilation=18)
        self.conv_1x1_output = ConvAltan(out_channels * 4, out_channels,True, kernel_size=1)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv_1x1_output(x_cat)
    

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
    