import os
import json
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "TempoFunk/webvid-10M"
NUM_ROWS = 100_000
INPUT_FILENAME = "batch_input.jsonl"
MISTRAL_MODEL = "mistral-large-latest" # Use a powerful model for this task

def create_llm_prompt(caption: str) -> str:
    """Creates the detailed prompt for the LLM."""
    return f"""
Analyze the following video caption and classify if it describes an action.

Your task is to:
1.  Carefully read the caption.
2.  Think step by step.
3.  Determine if it describes a distinct action (a subject performing a verb).
4.  If there is an action, rewrite the caption to be more descriptive, starting with "A video of...". Do not add new information.
5.  If there is no action (e.g., it's just a list of keywords, a title card, or a static description), classify it as "no_action".
6.  You MUST provide your response in the specified format below, including the <thinking> and <response> tags.

Caption to analyze: "{caption}"

---
Example 1 (Action):
Caption: "Funny naughty cat gnaws rope from the balloon."
<thinking>
The caption describes a clear action. The subject is "cat", the verb is "gnaws", and the object is "rope". I will classify this as 'action' and rewrite the sentence to start with "A video of...".
</thinking>
<response>
classification:action
rewritten_caption:A video of a funny, naughty cat gnawing on a rope from a balloon.
</response>

Example 2 (No Action):
Caption: "Michigan mi road map word travel tourism destination 3d animation"
<thinking>
This caption is a list of keywords and concepts related to a 3D animation. It does not describe a specific subject performing an action. I will classify this as 'no_action'.
</thinking>
<response>
classification:no_action
rewritten_caption:none
</response>
---

Now, provide your analysis for the caption provided above in the same format.
""".strip()

def prepare_batch_file():
    """Loads dataset and prepares the .jsonl input file for the Mistral Batch API."""
    print(f"Loading the first {NUM_ROWS} rows from '{DATASET_NAME}'...")
    # Using streaming=True is memory-efficient for large datasets
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    print(f"Creating '{INPUT_FILENAME}' for the batch job...")
    count = 0
    with open(INPUT_FILENAME, "w") as f:
        for row in dataset:
            if count >= NUM_ROWS:
                break

            caption = row["name"]
            videoid = row["videoid"]

            # The custom_id must be a unique string for each request.
            # The videoid is perfect for this.
            request = {
                "custom_id": str(videoid),
                "body": {
                    "model": MISTRAL_MODEL,
                    "messages": [{"role": "user", "content": create_llm_prompt(caption)}],
                    "max_tokens": 256, # Should be enough for the response format
                },
                # The endpoint this request should be sent to
                "endpoint": "/v1/chat/completions",
            }
            f.write(json.dumps(request) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  ...processed {count}/{NUM_ROWS} rows")

    print(f"Successfully created '{INPUT_FILENAME}' with {count} requests.")

def main():
    """Main function to run the script."""
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env file")

    # 1. Prepare the batch input file
    prepare_batch_file()

    # 2. Initialize Mistral client
    client = Mistral(api_key=api_key)

    # 3. Upload the batch file
    print(f"\nUploading '{INPUT_FILENAME}' to Mistral...")
    # CORRECTION: The 'file' argument must be a dictionary with 'file_name' and 'content' keys.
    with open(INPUT_FILENAME, "rb") as f_content:
        batch_file = client.files.upload(
            file={"file_name": INPUT_FILENAME, "content": f_content},
            purpose="batch"
        )
    print(f"File uploaded successfully. File ID: {batch_file.id}")

    # 4. Create and launch the batch job
    print("Creating batch job...")
    created_job = client.batch.jobs.create(
        input_files=[batch_file.id],
        model=MISTRAL_MODEL, # This model is used if not specified in the individual request body
        endpoint="/v1/chat/completions",
        metadata={"job_type": "webvid_caption_classification"}
    )
    print("\n--- Batch Job Submitted ---")
    print(f"Job ID: {created_job.id}")
    print(f"Status: {created_job.status}")
    print("\nThe job is now running. It may take a significant amount of time.")
    print("Run the `2_check_and_process.py` script periodically to check status and process results when complete.")

if __name__ == "__main__":
    main()
