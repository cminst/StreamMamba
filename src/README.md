# Installation

## Requirements

```shell
~/StreamMamba/mamba$ pip install -e .
~/StreamMamba/src$ pip install -r requirements.txt
```

In addition, to ensure no issues arise with installation of Flash Attention, feel free to install prebuilt wheels from [here](https://github.com/mjun0812/flash-attention-prebuild-wheels)

## Note

Before running any training scripts, set the environment variable `DATASET_ROOT`

## Key Dependencies Installation for FlashAttention2

Some modules (FusedMLP and DropoutLayerNorm) from FlashAttention2 used in our models rely on CUDA extensions.

1. Prerequisite for installation: Refer to the [requirements](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) in flash-attention.
2. Clone [flash-attention](https://github.com/Dao-AILab/flash-attention) project or download its code to your machine. Change current directory to flash-attention: ````cd flash-attention``.
3. Install fused_mlp_lib. Refer to [here](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/fused_dense_lib).
```python
cd csrc/fused_dense_lib && pip install .
```
4. Install layer_form. Refer to [here](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm).
```python
cd csrc/layer_norm && pip install .
```
