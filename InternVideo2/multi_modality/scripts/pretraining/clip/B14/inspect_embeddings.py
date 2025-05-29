import os
import logging
from os.path import join

import torch
from tqdm import tqdm

logging.basicConfig(level = logging.INFO)

logger = logging.getLogger(__name__)

get_file = lambda step: f"B14/kinetics-embeddings/step-{step}/embeddings.pt"

def main():
    step = 0

    while os.path.exists(get_file(step)):
        window_tensors = torch.load(get_file(step))

        logger.info(f"For step {step}, the tensor keys are {list(window_tensors.keys())}")
        logger.info(f"For step {step}, the embedding shape is {window_tensors[list(window_tensors.keys())[0]].shape}")

        if step == 5:
            logger.info("Breaking...")
            break

        del window_tensors

    logger.info("Done!")

if __name__ == '__main__':
    logger.info("Started inspection of embeddings!")

    main()
