import logging
import torch
import json
import os
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
import numpy as np

# --- Configuration ---
CLASSIFICATION_MODEL = "./ActionBert2-109M/"
GENERATION_MODEL = "./ReAction-1.5B/"
DATASET_ID = "TempoFunk/webvid-10M"
OUTPUT_HF_REPO = "qingy2024/webvid-10M-pro-scored"
CLASSIFICATION_OUTPUT_FILE = "webvid_scores.jsonl"

# --- New Configuration ---
# The number of top-scoring captions to select for rewriting
TOP_N_FOR_GENERATION = 100_000

CLASSIFICATION_BATCH_SIZE = 1024
GENERATION_BATCH_SIZE = 900

PROMPT_TEMPLATE = """<|im_start|>system
You are ReAction, an assistant to rewrite video caption to make them clearer. Rewrite the user's video caption.<|im_end|>
<|im_start|>user
{caption_text_from_name_column}<|im_end|>
<|im_start|>assistant
"""

def main():
    """Main function to run the processing pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting WebVid caption scoring and rewriting script.")

    logger.info(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split='train')
    logger.info(f"Dataset loaded with {len(dataset)} rows.")

    scores = []

    if os.path.exists(CLASSIFICATION_OUTPUT_FILE):
        logger.info(f"Found existing scoring file: {CLASSIFICATION_OUTPUT_FILE}")
        logger.info("Loading scores from file...")

        try:
            with open(CLASSIFICATION_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading scores"):
                    data = json.loads(line.strip())
                    scores.append(data['score'])

            if len(scores) != len(dataset):
                logger.warning(f"Number of scores ({len(scores)}) doesn't match dataset size ({len(dataset)})")
                logger.info("Re-running scoring...")
                scores = []
            else:
                logger.info(f"Successfully loaded {len(scores)} scores.")

        except Exception as e:
            logger.error(f"Error loading scoring file: {e}")
            logger.info("Re-running scoring...")
            scores = []
    else:
        logger.info(f"No existing scoring file found. Running scoring...")

    # Only run scoring if we don't have valid scores
    if not scores:
        logger.info(f"Initializing scoring model with pipeline: {CLASSIFICATION_MODEL}")
        classifier = pipeline(
            "text-classification",
            model=CLASSIFICATION_MODEL,
            tokenizer=CLASSIFICATION_MODEL,
            device='cuda:0',
            batch_size=CLASSIFICATION_BATCH_SIZE,
            torch_dtype=torch.bfloat16 # Use bfloat16 for better performance
        )

        logger.info("Starting caption scoring using the pipeline...")

        with open(CLASSIFICATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # Use KeyDataset for highly efficient iteration with the pipeline
            dataset_iterator = KeyDataset(dataset, "name")

            progress_bar = tqdm(
                enumerate(classifier(dataset_iterator)),
                total=len(dataset),
                desc="Scoring captions"
            )

            for idx, result in progress_bar:
                score = result['score']
                scores.append(score)

                # Save score result to JSONL file for caching
                score_entry = {
                    "index": idx,
                    "caption": dataset[idx]["name"],
                    "score": score
                }
                f.write(json.dumps(score_entry) + '\n')

        logger.info(f"Scoring results saved to {CLASSIFICATION_OUTPUT_FILE}")
        logger.info("Scoring complete.")

    # Add score column to the dataset
    logger.info("Adding scores to the dataset.")
    dataset = dataset.add_column("action_score", scores)

    # Sort the dataset by the new score column in descending order
    logger.info("Sorting dataset by action score...")
    sorted_dataset = dataset.sort("action_score", reverse=True)

    # Determine how many items to select, ensuring we don't exceed dataset size
    num_to_rewrite = min(TOP_N_FOR_GENERATION, len(sorted_dataset))
    logger.info(f"Selecting top {num_to_rewrite} captions for rewriting.")

    # Split dataset into the top N and the rest
    top_n_subset = sorted_dataset.select(range(num_to_rewrite))
    rest_subset = sorted_dataset.select(range(num_to_rewrite, len(sorted_dataset)))

    if len(top_n_subset) > 0:
        logger.info(f"Initializing generation model with vLLM: {GENERATION_MODEL}")
        llm = LLM(
            model=GENERATION_MODEL,
            tensor_parallel_size=1,
            dtype='bfloat16',
            max_model_len=1024,
            gpu_memory_utilization=0.9,
        )
        sampling_params = SamplingParams(
            temperature=0.3, top_p=0.8, top_k=20,
            repetition_penalty=1.05, max_tokens=256,
            skip_special_tokens=True,
        )

        logger.info(f"Generating rewritten captions for {len(top_n_subset)} top-scored captions...")
        rewritten_captions = []

        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(top_n_subset), GENERATION_BATCH_SIZE), desc="Generating captions"):
            batch = top_n_subset[i:i + GENERATION_BATCH_SIZE]
            prompts = [PROMPT_TEMPLATE.format(caption_text_from_name_column=c) for c in batch["name"]]
            outputs = llm.generate(prompts, sampling_params)
            batch_captions = [output.outputs[0].text.strip() for output in outputs]
            rewritten_captions.extend(batch_captions)

        top_n_subset = top_n_subset.add_column("rewritten_caption", rewritten_captions)
        logger.info("Generation complete.")
    else:
        logger.warning("No captions selected for rewriting.")

    logger.info(f"Adding 'none' as rewritten_caption for the remaining {len(rest_subset)} captions.")
    if len(rest_subset) > 0:
        rest_subset = rest_subset.add_column("rewritten_caption", ["none"] * len(rest_subset))
    else:
        # Handle edge case where all items are selected for rewriting
        if len(top_n_subset) > 0:
            rest_subset = Dataset.from_dict({col: [] for col in top_n_subset.column_names})

    logger.info("Combining top-scored and remaining subsets.")
    if len(top_n_subset) > 0 and len(rest_subset) > 0:
        final_dataset = concatenate_datasets([top_n_subset, rest_subset])
    elif len(top_n_subset) > 0:
        final_dataset = top_n_subset
    else:
        final_dataset = rest_subset

    logger.info(f"Pushing final dataset ({len(final_dataset)} rows) to HF Hub: {OUTPUT_HF_REPO}")
    final_dataset.push_to_hub(OUTPUT_HF_REPO, private=True)

    logger.info("Script completed successfully.")



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
