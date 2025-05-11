# Tintelligence

# 🌀 Minimal DDPM Sampler in PyTorch

This repository contains a minimal, yet complete, implementation of a Denoising Diffusion Probabilistic Model (DDPM) built with PyTorch. The project includes:

- A custom `DDPMSampler` with configurable training and inference steps.
- Built-in grayscale image support with optional 3-channel output.
- Flexible noise scheduling and backward sampling using DDPM equations.
- Ready-to-use for generating synthetic data or integrating with your own UNet/Transformer models.

---

## 🚀 Features

- ✅ Clean, modular DDPM sampler
- 🧠 Supports training steps (e.g., `num_training_steps=200`)
- 🔍 Inference with custom step size (e.g., `num_inference_steps=50`)
- 🎨 Grayscale image preprocessing (`transforms.Grayscale(num_output_channels=3)`)
- 📉 Supports `add_noise()` and reverse `step()` functions as in DDPM papers

---

## 📚 Paper Reference

- "Denoising Diffusion Probabilistic Models" by Ho et al. (https://arxiv.org/abs/2006.11239)
