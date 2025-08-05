# StreamMamba: Self-Predictive Frame Skipping for Real-Time Prompt-Based Peak Frame Detection

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains the official PyTorch implementation for the paper: **"StreamMamba: Self-Predictive Frame Skipping for Real-Time Prompt-Based Peak Frame Detection"**.

StreamMamba introduces a novel framework for efficient video understanding that dramatically reduces computational cost without significant loss in performance. It leverages a Mamba state-space model trained to predict future video content, allowing it to dynamically skip processing redundant frames. This makes it ideal for real-time applications on resource-constrained devices like smartphones.

**Note:** The `mamba` folder in this repository is copied from the original implementation at https://github.com/state-spaces/mamba.

## Training and Inference Instructions

For detailed instructions on how to prepare datasets and models, as well as how to run training and inference, please refer to:

- [Dataset and Model Preparation](src/DATASET_MODELS.md)
- [Installation and Usage](src/README.md)
