# Dataset Preparation

# Kinetics-600 Subset (Slim-Kinetics-2)

The `slim_kinetics` dataset can be downloaded from [qingy2024/Slim-Kinetics-2](https://huggingface.co/datasets/qingy2024/Slim-Kinetics-2) on HuggingFace:

```
huggingface-cli download qingy2024/Slim-Kinetics-2 --local-dir $DATASET_ROOT --repo-type=dataset
```


# Model Preparation

Model files can be found at [qingy2024/InternVideo2-B14](https://huggingface.co/qingy2024/InternVideo2-B14/tree/main), although the model checkpoints are automatically downloaded in the training scripts.
