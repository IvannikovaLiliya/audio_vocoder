# Fast audio generation by Deep Learning with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving Vocoder task with PyTorch.

## Installation

Follow these steps to install the project:

0. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=python3.11

   # activate env
   conda activate project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install the best model checkpoint

   ```bash
   sh cp_model_fin/model_fin
   ```

## Inference of model

   python inference/inference.py

## Train the model

   python trainer/train_fin.py -cfg_filename configs/model_fin.json

## Experimets

   [Vocoder Project](https://api.wandb.ai/links/lo-ivannikova-hse/qkykibtj)
