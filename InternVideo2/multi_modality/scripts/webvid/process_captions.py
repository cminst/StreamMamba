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

DATASET_NAME = "TempoFunk/webvid-10M"
NUM_ROWS_TO_PROCESS = 100_000
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
HF_HUB_REPO_ID = "qingy2024/webvid-10M-scored"

MAX_CONCURRENT_REQUESTS = 20
MAX_TOKENS_PER_MINUTE = 2_000_000

# Retry Configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY_S = 2 # Initial delay in seconds for the first retry

# File Configuration
PROGRESS_FILE = "progress.jsonl"

# Rate Limiter and Retry Utilities

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


# LLM Prompt and Parsing

def create_llm_prompt(caption: str) -> str:
    """Creates the detailed prompt for the LLM to rate action intensity."""
    return f"""
Analyze the following video caption and rate its "action intensity" on a scale from 0 to 5. The key factor is the presence of a clear, dynamic "peak moment" of action.

Your task is to:
1.  Carefully read the caption.
2.  Think step-by-step inside <thinking> tags about the type of action described.
3.  Assign an `action_score` from 0 to 5 based on the scale below.
4.  If the score is 1 or higher, rewrite the caption to be more descriptive, starting with "A video of...". Do not add new information.
5.  If the score is 0, the rewritten caption should be `none`.
6.  You MUST provide your response in the specified format, including the <thinking> and <response> tags.

**Action Score Scale:**
*   **0 (Static/Scenic):** No action or subject. Describes a static scene, a location, or is just a list of keywords.
    (e.g., "A mushroom in a forest", "New York City skyline", "background animation texture")
*   **1 (Ambient/Subtle Motion):** Very low-energy, passive, or ambient movement. No clear subject performing an intentional action.
    (e.g., "Clouds drifting in the sky", "Water gently rippling", "Sunlight shining through branches")
*   **2 (Low-Energy/Sustained Action):** A clear action is present, but it's calm, sustained, or lacks a distinct peak.
    (e.g., "A person sitting on a bench and reading a book", "Kids playing with toys on the floor", "Someone walking down a street")
*   **3 (Moderate Action):** A more dynamic or purposeful action that has energy but may not have a single, dramatic peak.
    (e.g., "A group of people dancing", "A car driving on a highway", "A chef chopping vegetables quickly")
*   **4 (High-Energy Action/Clear Peak):** A high-energy action with a clear and distinct peak moment. The action builds to or completes a significant, visually interesting point.
    (e.g., "A whale jumping out of the water", "A soccer player scoring a goal", "A gymnast landing a routine")
*   **5 (Peak Moment/Climactic Action):** The caption describes the absolute climax or the most intense, singular moment of an action. This is the apex.
    (e.g., "A person performs a backflip, midair", "A lightning strike illuminating the sky", "An explosion erupting")

Caption to analyze: "{caption}"

---
**Example 1 (Peak Moment):**
Caption: "Man does a backflip on a trampoline"
<thinking>
The caption describes a very high-energy, specific action: a backflip. This action has a clear and dramatic peak moment when the person is inverted in the air. This fits the definition of a score 5, as it's the climax of the action. I will rewrite the caption.
</thinking>
<response>
action_score:5
rewritten_caption:A video of a man doing a backflip on a trampoline.
</response>

**Example 2 (Sustained Action):**
Caption: "woman sitting on a park bench reading"
<thinking>
The caption describes an action (reading), but it is a calm, low-energy, and sustained activity. There is no peak moment or high drama. This perfectly matches the description for a score of 2. I will rewrite the caption.
</thinking>
<response>
action_score:2
rewritten_caption:A video of a woman sitting on a park bench and reading a book.
</response>

**Example 3 (Static Scene):**
Caption: "A beautiful mushroom in the forest"
<thinking>
This caption describes a static object in a scene. There is no subject performing any action. This is a classic example of a score 0. The rewritten caption will be 'none'.
</thinking>
<response>
action_score:0
rewritten_caption:none
</response>

**Example 4 (High-Energy with Peak):**
Caption: "A whale breaches the surface of the ocean"
<thinking>
The caption describes a powerful, high-energy event. A whale breaching has a very clear peak moment as it emerges from the water. This is a strong example of a score 4. I will rewrite the caption.
</thinking>
<response>
action_score:4
rewritten_caption:A video of a whale breaching the surface of the ocean.
</response>
---

Now, provide your analysis for the caption provided above in the same format.
""".strip()

def parse_llm_response(content: str) -> dict:
    """Parses the structured response from the LLM, returning a dictionary."""
    try:
        response_match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        if not response_match:
            print(f"Parse Error: No <response> tag found in content: {content}")
            return {"action_score": -1, "rewritten_caption": "no_response_tag"}

        response_text = response_match.group(1).strip()

        action_score = -1 # Default to -1 for error
        rewritten_caption = "missing_caption"

        score_match = re.search(r"action_score:\s*(\d+)", response_text)
        if score_match:
            try:
                # Convert the extracted score to an integer
                action_score = int(score_match.group(1).strip())
            except (ValueError, TypeError):
                print(f"Parse Error: Could not convert action_score to int in response: {response_text}")
                action_score = -1 # Mark as error if not a valid integer
        else:
            rewritten_caption = "missing_action_score"


        caption_match = re.search(r"rewritten_caption:(.*)", response_text, re.DOTALL)
        if caption_match:
            rewritten_caption = caption_match.group(1).strip()

        return {"action_score": action_score, "rewritten_caption": rewritten_caption}
    except Exception as e:
        return {"action_score": -1, "rewritten_caption": str(e)}

# Asynchronous Processing Core

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

            llm_output = response.choices[0].message.content
            parsed_data = parse_llm_response(llm_output)

            # Record token usage on success
            tokens_used = 0
            try:
                tokens_used = response.usage.total_tokens
                await token_limiter.add_usage(tokens_used)
            except (KeyError, IndexError, TypeError):
                print(f"Warning: Could not extract token count for videoid {videoid}. Usage not recorded.")

            # Log result
            result = {
                "videoid": videoid,
                "original_caption": caption,
                "action_score": parsed_data["action_score"],
                "rewritten_caption": parsed_data["rewritten_caption"],
                "tokens_used": tokens_used,
                "status": "success"
            }

        except Exception as e:
            result = {
                "videoid": videoid,
                "original_caption": caption,
                "action_score": -1, # Use -1 to indicate an API or other error
                "rewritten_caption": str(e),
                "tokens_used": 0,
                "status": "error"
            }

        async with aiofiles.open(PROGRESS_FILE, 'a') as f:
            await f.write(json.dumps(result) + '\n')

        pbar.update(1)
        return result


async def main():
    """Main function to orchestrate the entire process."""
    load_dotenv()

    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    if not nebius_api_key or not hf_token:
        raise ValueError("NEBIUS_API_KEY or HF_TOKEN not found in .env file.")

    client = AsyncOpenAI(
        api_key=nebius_api_key,
        base_url="https://api.studio.nebius.com/v1",
    )

    print("---- Configuration ----")
    print(f"Model: {LLM_MODEL}")
    print(f"Repo ID: {HF_HUB_REPO_ID}")
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