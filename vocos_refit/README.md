# Vocos refitting LJSpeech by Deep Learning with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for refitting Vocos task with PyTorch.

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
   pip install -r requirements-train.txt
   ```

## Train the model

   python train.py -c configs/vocos.yaml
