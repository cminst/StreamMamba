# Dataset Preparation

## Kinetics-600 Subset (Slim-Kinetics-2)

The `slim_kinetics` dataset can be downloaded from [cminst/Slim-Kinetics-2](https://huggingface.co/datasets/cminst/Slim-Kinetics-2) on HuggingFace:

```
huggingface-cli download cminst/Slim-Kinetics-2 --local-dir $DATASET_ROOT --repo-type=dataset
```

## Adding Custom Datasets

To add your own custom dataset, follow these steps:

1. Define your dataset in `configs/data.py` within the `available_corpus` dictionary
2. Create a dictionary entry with a unique name for your dataset
3. Specify the following required fields:
   - `anno_path`: Path to your annotation file (JSON format)
   - `data_root`: Root directory containing your media files
   - `media_type`: Either "image" or "video"

Example for a custom video dataset:
```python
custom_video_dataset=dict(
    anno_path="/path/to/your/annotations.json",
    data_root="/path/to/your/videos/",
    media_type="video",
    min_caption_length=1  # Optional: minimum caption length filter
)
```

Example for a custom image dataset:
```python
custom_image_dataset=dict(
    anno_path="/path/to/your/annotations.json",
    data_root="/path/to/your/images/",
    media_type="image"
)
```

4. Reference your dataset in training scripts using the name you assigned (e.g., `custom_video_dataset`)

# Model Preparation

Model files can be found at [cminst/InternVideo2-B14](https://huggingface.co/cminst/InternVideo2-B14/tree/main), although the model checkpoints are automatically downloaded in the training scripts.
