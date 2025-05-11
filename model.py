import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.unet import UNet
from components.vae_encoder import VAEencoder
from components.vae_decoder import Vaedecoder
from components.ddpm import DDPMSampler
from tqdm import tqdm
import gc
gc.collect()

# PyTorch GPU belleğini boşalt
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
device = "cuda"

class Tintelligence (nn.Module):
    """Some Information Model"""
    def __init__(self,sampler_name="ddpm",n_inference_steps=50,seed=None):
        super().__init__()
        self.generator = torch.Generator(device=device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)
        self.strength=0.4
        self.encoder=VAEencoder().to(device)
        self.unet=UNet().to(device)
        self.decoder=Vaedecoder().to(device)

        if sampler_name == "ddpm":
            self.sampler = DDPMSampler(self.generator)
            self.sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        
    def forward(self, x):
        B,C,H,W=x.shape
        latents_shape = (1, 4, H//8,W//8)
        noise = torch.rand(latents_shape,device=device ,generator=self.generator)
        x=self.encoder(x,noise)
        self.sampler.set_strength(strength=self.strength)
        x = self.sampler.add_noise(x, self.sampler.timesteps[0])
        timesteps = tqdm(self.sampler.timesteps)
        for i,timestep in enumerate(timesteps):
           
            model_output=self.unet(x,timestep)

            x=self.sampler.step(timestep,x,model_output)
        
        x=self.decoder(x)

        return x
if __name__=="__main__":
    resim=torch.rand(1,3,128,128).to(device)
    model=Tintelligence().to(device)
    output=model(resim)
    print(output.shape)
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Toplam Parametre Sayısı     : {total_params:,}")
        print(f"Eğitilebilir Parametre Sayısı : {trainable_params:,}")
        return total_params, trainable_params
    
    count_parameters(model)