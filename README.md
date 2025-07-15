# StreamMamba: Efficient Video Understanding through Adaptive Computation

[![Paper](https://img.shields.io/badge/arxiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/[TODO: ADD_ARXIV_ID_WHEN_AVAILABLE])
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
`[TODO: Add other badges like a Hugging Face Spaces demo if you create one]`

This repository contains the official PyTorch implementation for the paper: **"StreamMamba: Adaptive Computation for Efficient Video State-Space Models"**.

StreamMamba introduces a novel framework for efficient video understanding that dramatically reduces computational cost without significant loss in performance. It leverages a Mamba state-space model trained to predict future video content, allowing it to dynamically skip processing redundant frames. This makes it ideal for real-time applications on resource-constrained devices like smartphones.
