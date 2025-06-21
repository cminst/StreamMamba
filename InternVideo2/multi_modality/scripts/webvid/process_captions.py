import os
import json
import re
import time
import asyncio
import random
from collections import deque
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm
import aiofiles

# --- Configuration ---
# -----------------------------------------------------------------------------
# Dataset and Model Configuration
DATASET_NAME = "TempoFunk/webvid-10M"
NUM_ROWS_TO_PROCESS = 100_000
LLM_MODEL = "Llama-3.3-70B-Instruct"  # Or another suitable model
HF_HUB_REPO_ID = "qingy2024/webvid-10M-classified" # <<<--- CHANGE THIS

# API and Rate Limiting Configuration
# Meta's default is 3000 RPM. 3000/60 = 50 RPS.
# We set max concurrent requests slightly lower to be safe.
MAX_CONCURRENT_REQUESTS = 30
# New: Token-per-minute limit for the Llama.com API
MAX_TOKENS_PER_MINUTE = 900_000 # Set slightly below the 1M limit for safety

# Retry Configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY_S = 2 # Initial delay in seconds for the first retry

# File Configuration
PROGRESS_FILE = "progress.jsonl"

# --- NEW: Rate Limiter and Retry Utilities ---
# -----------------------------------------------------------------------------

class AsyncTokenRateLimiter:
    """
    Manages token usage to stay within a rate limit over a rolling time window.
    This is essential for APIs with token-per-minute limits.
    """
    def __init__(self, max_tokens: int, time_window_seconds: int = 60):
        self.max_tokens = max_tokens
        self.time_window_seconds = time_window_seconds
        self.lock = asyncio.Lock()
        self.history = deque() # Stores (timestamp, token_count) tuples
        self.current_tokens = 0

    def _prune(self):
        """Removes entries from history that are older than the time window."""
        now = time.monotonic()
        while self.history and self.history[0][0] < now - self.time_window_seconds:
            self.current_tokens -= self.history.popleft()[1]

    async def add_usage(self, tokens_used: int):
        """Records token usage for a completed request."""
        async with self.lock:
            self._prune()
            timestamp = time.monotonic()
            self.history.append((timestamp, tokens_used))
            self.current_tokens += tokens_used

    async def wait_for_slot(self):
        """
        Waits until there is capacity for a new request under the token limit.
        This should be called *before* making an API call.
        """
        while True:
            async with self.lock:
                self._prune()
                if self.current_tokens < self.max_tokens:
                    # There is enough capacity, we can proceed
                    return

                # Calculate wait time needed for the oldest request to expire
                oldest_timestamp = self.history[0][0]
                wait_time = (oldest_timestamp + self.time_window_seconds) - time.monotonic()
                wait_time = max(0, wait_time) + 0.1 # Add a small buffer

            # Release the lock and wait
            await asyncio.sleep(wait_time)


def async_retry(max_retries=3, initial_delay=1, backoff_factor=2,
                exceptions_to_retry=(RateLimitError, APITimeoutError, APIConnectionError)):
    """
    A decorator for retrying an async function with exponential backoff.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions_to_retry as e:
                    if attempt == max_retries:
                        print(f"Final attempt failed. Re-raising exception: {e}")
                        raise

                    jitter = random.uniform(0, delay * 0.1) # Add jitter to avoid thundering herd
                    sleep_time = delay + jitter

                    print(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {sleep_time:.2f} seconds...")

                    await asyncio.sleep(sleep_time)
                    delay *= backoff_factor
        return wrapper
    return decorator


# --- LLM Prompt and Parsing (Unchanged) ---
# -----------------------------------------------------------------------------

def create_llm_prompt(caption: str) -> str:
    """Creates the detailed prompt for the LLM."""
    return f"""
Analyze the following video caption and classify if it describes an action.

Your task is to:
1.  Carefully read the caption.
2.  Think step by step inside <thinking> tags.
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

def parse_llm_response(content: str) -> dict:
    """Parses the structured response from the LLM, returning a dictionary."""
    try:
        response_match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        if not response_match:
            print(f"Error: {content}")
            return {"classification": "parse_error", "rewritten_caption": "no_response_tag"}

        response_text = response_match.group(1).strip()

        classification = "parse_error"
        rewritten_caption = "missing_caption"

        class_match = re.search(r"classification:(.*)", response_text)
        if class_match:
            classification = class_match.group(1).strip()

        caption_match = re.search(r"rewritten_caption:(.*)", response_text, re.DOTALL)
        if caption_match:
            rewritten_caption = caption_match.group(1).strip()

        return {"classification": classification, "rewritten_caption": rewritten_caption}
    except Exception as e:
        return {"classification": "parse_error", "rewritten_caption": str(e)}

# --- Asynchronous Processing Core (Updated) ---
# -----------------------------------------------------------------------------

# Apply the retry decorator to the processing function
@async_retry(max_retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY_S)
async def process_item(item, client, semaphore, pbar, token_limiter):
    """Processes a single item: waits for limits, calls API, and parses response."""
    async with semaphore:
        # Wait for both the token-per-minute slot and the concurrent request slot
        await token_limiter.wait_for_slot()

        videoid = item["videoid"]
        caption = item["name"]

        try:
            prompt = create_llm_prompt(caption)
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                temperature=0.6,
                max_completion_tokens=4096,
                top_p=0.9,
            )

            llm_output = response.completion_message["content"]["text"]
            parsed_data = parse_llm_response(llm_output)

            # Record token usage on success
            tokens_used = 0
            try:
                # Using the specific metric structure for Llama.com API
                tokens_used = int(response.metrics[-1]['value'])
                await token_limiter.add_usage(tokens_used)
            except (KeyError, IndexError, TypeError):
                print(f"Warning: Could not extract token count for videoid {videoid}. Usage not recorded.")


            result = {
                "videoid": videoid,
                "original_caption": caption,
                "classification": parsed_data["classification"],
                "rewritten_caption": parsed_data["rewritten_caption"],
                "tokens_used": tokens_used, # Optional: good to log
                "status": "success"
            }

        except Exception as e:
            # This block now catches errors that persist *after* all retries
            result = {
                "videoid": videoid,
                "original_caption": caption,
                "classification": "api_error",
                "rewritten_caption": str(e),
                "tokens_used": 0,
                "status": "error"
            }

        async with aiofiles.open(PROGRESS_FILE, 'a') as f:
            await f.write(json.dumps(result) + '\n')

        pbar.update(1) # Manually update the progress bar on completion
        return result

# --- Main Execution (Updated) ---
# -----------------------------------------------------------------------------

async def main():
    """Main function to orchestrate the entire process."""
    load_dotenv()

    llama_api_key = os.getenv("LLAMA_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    if not llama_api_key or not hf_token:
        raise ValueError("LLAMA_API_KEY or HF_TOKEN not found in .env file.")

    client = AsyncOpenAI(
        api_key=llama_api_key,
        base_url="https://api.llama.com/v1",
    )

    print("---- Configuration ----")
    print(f"Model: {LLM_MODEL}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Max Tokens per Minute: {MAX_TOKENS_PER_MINUTE:,}")
    print(f"Max Retries: {MAX_RETRIES}")
    print("-----------------------")

    print("\nLoading dataset and checking for existing progress...")
    processed_ids = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'videoid' in data:
                        processed_ids.add(data['videoid'])
                except json.JSONDecodeError:
                    print(f"Skipping corrupted line in progress file: {line.strip()}")
        print(f"Found {len(processed_ids)} already processed items. Resuming...")

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    items_to_process = []
    # Take a fixed number of items to build the list
    for item in dataset.take(NUM_ROWS_TO_PROCESS):
        if item['videoid'] not in processed_ids:
            items_to_process.append(item)

    if not items_to_process:
        print("All items have already been processed. Moving to finalization.")
    else:
        print(f"Total items to process in this run: {len(items_to_process)}")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        # Instantiate the token rate limiter
        token_limiter = AsyncTokenRateLimiter(max_tokens=MAX_TOKENS_PER_MINUTE)

        # We manually create the progress bar and pass it to the worker
        with tqdm(total=len(items_to_process), desc="Processing Captions", unit="caption") as pbar:
            # Pass the token_limiter to each task
            tasks = [process_item(item, client, semaphore, pbar, token_limiter) for item in items_to_process]
            await asyncio.gather(*tasks)

        print("\nAll items processed.")

    print("\nFinalizing results and preparing for upload...")
    final_results = []
    with open(PROGRESS_FILE, 'r') as f:
        for line in f:
             try:
                final_results.append(json.loads(line))
             except json.JSONDecodeError:
                print(f"Skipping corrupted line during finalization: {line.strip()}")

    final_dataset = Dataset.from_list(final_results)

    print(f"Uploading dataset to Hugging Face Hub at '{HF_HUB_REPO_ID}'...")
    final_dataset.push_to_hub(HF_HUB_REPO_ID, token=hf_token, private=True)

    print("\n--- All Done! ---")
    print(f"Visit your dataset at: https://huggingface.co/datasets/{HF_HUB_REPO_ID}")

if __name__ == "__main__":
    asyncio.run(main())
