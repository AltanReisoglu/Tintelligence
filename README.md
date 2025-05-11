# 🌀 Tintelligence

This project implements a basic **DDPM (Denoising Diffusion Probabilistic Model)** . It is designed to be simple, readable, and customizable for experimentation or educational purposes.

## 🔧 Features

- Minimal PyTorch implementation of DDPM  
- Configurable training (`num_training_steps`) and inference (`num_inference_steps`) steps  
- Supports noise addition and reverse denoising  
- Easily integrates with your own UNet-like models

## 🚀 Usage

```python
from ddpm import DDPMSampler
import torch

generator = torch.manual_seed(42)
sampler = DDPMSampler(generator=generator, num_training_steps=200)
sampler.set_inference_timesteps(num_inference_steps=50)

# Example sampling loop
latents = torch.randn((1, 3, 64, 64))  # random noise
for timestep in sampler.timesteps:
    model_output = my_model(latents, timestep)  # replace with your model
    latents = sampler.step(timestep, latents, model_output)
