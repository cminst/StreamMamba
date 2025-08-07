import logging
import torch
import json
import os
from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
import numpy as np

CLASSIFICATION_MODEL = "./action_classifier/"
GENERATION_MODEL = "./ReAction-1.5B/"
DATASET_ID = "TempoFunk/webvid-10M"
OUTPUT_HF_REPO = "qingy2024/webvid-10M-pro"
CLASSIFICATION_OUTPUT_FILE = "webvid_classification.jsonl"

CLASSIFICATION_BATCH_SIZE = 1024
GENERATION_BATCH_SIZE = 900

PROMPT_TEMPLATE = """{caption_text_from_name_column}"""

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting WebVid caption classification and rewriting script.")

    logger.info(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split='train')
    logger.info(f"Dataset loaded with {len(dataset)} rows.")

    if os.path.exists(CLASSIFICATION_OUTPUT_FILE):
        logger.info(f"Found existing classification file: {CLASSIFICATION_OUTPUT_FILE}")
        logger.info("Loading classifications from file.")

        classifications = []
        action_count = 0
        no_action_count = 0

        try:
            with open(CLASSIFICATION_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading classifications"):
                    data = json.loads(line.strip())
                    classification = data['classification']
                    classifications.append(classification)

                    if classification == 'action':
                        action_count += 1
                    elif classification == 'no_action':
                        no_action_count += 1

            if len(classifications) != len(dataset):
                logger.warning(f"Number of classifications ({len(classifications)}) doesn't match dataset size ({len(dataset)})")
                logger.info("Re-running classification.")
                classifications = []
            else:
                logger.info(f"Successfully loaded {len(classifications)} classifications")
                logger.info(f"Found {action_count} 'action' captions and {no_action_count} 'no_action' captions.")

        except Exception as e:
            logger.error(f"Error loading classification file: {e}")
            logger.info("Re-running classification.")
            classifications = []

    else:
        logger.info("No existing classification file found. Running classification.")
        classifications = []

    if not classifications:
        logger.info(f"Initializing classification model: {CLASSIFICATION_MODEL}")
        classifier = pipeline(
            "text-classification",
            model=CLASSIFICATION_MODEL,
            tokenizer=CLASSIFICATION_MODEL,
            device='cuda:0',
            batch_size=CLASSIFICATION_BATCH_SIZE
        )

        logger.info("Starting caption classification.")

        classifications = []
        action_count = 0
        no_action_count = 0

        with open(CLASSIFICATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            progress_bar = tqdm(
                enumerate(classifier(KeyDataset(dataset, "name"))),
                total=len(dataset),
                desc="Classifying captions"
            )

            for idx, result in progress_bar:
                label = result['label']
                classifications.append(label)

                classification_entry = {
                    "index": idx,
                    "caption": dataset[idx]["name"],
                    "classification": label,
                    "confidence": result['score']
                }
                f.write(json.dumps(classification_entry) + '\n')

                if label == 'action':
                    action_count += 1
                elif label == 'no_action':
                    no_action_count += 1

                progress_bar.set_postfix(actions=action_count, no_actions=no_action_count)

        logger.info(f"Classification results saved to {CLASSIFICATION_OUTPUT_FILE}")
        logger.info("Classification complete.")
        logger.info(f"Found {action_count} 'action' captions and {no_action_count} 'no_action' captions.")

    dataset = dataset.add_column("classification", classifications)

    action_subset = dataset.filter(lambda x: x['classification'] == 'action')
    no_action_subset = dataset.filter(lambda x: x['classification'] == 'no_action')

    if len(action_subset) > 0:
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

        logger.info(f"Generating rewritten captions for {len(action_subset)} action captions.")
        rewritten_captions = []

        for i in tqdm(range(0, len(action_subset), GENERATION_BATCH_SIZE), desc="Generating captions"):
            batch = action_subset[i:i + GENERATION_BATCH_SIZE]
            prompts = [PROMPT_TEMPLATE.format(caption_text_from_name_column=c) for c in batch["name"]]
            outputs = llm.generate(prompts, sampling_params)
            batch_captions = [output.outputs[0].text.strip() for output in outputs]
            rewritten_captions.extend(batch_captions)

        action_subset = action_subset.add_column("rewritten_caption", rewritten_captions)
        logger.info("Generation complete.")

    logger.info("Adding 'none' as rewritten_caption for no-action subset.")
    no_action_subset = no_action_subset.add_column("rewritten_caption", ["none"] * len(no_action_subset))

    logger.info("Combining action and no-action subsets.")
    final_dataset = concatenate_datasets([action_subset, no_action_subset])

    logger.info(f"Pushing final dataset ({len(final_dataset)} rows) to HF Hub: {OUTPUT_HF_REPO}")
    final_dataset.push_to_hub(OUTPUT_HF_REPO, private=True)

    logger.info("Script completed successfully.")


if __name__ == '__main__':
    main()
