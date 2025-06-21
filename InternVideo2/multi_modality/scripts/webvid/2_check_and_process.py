import os
import json
import re
import time
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from datasets import Dataset
from huggingface_hub import HfApi

# --- Configuration ---
OUTPUT_FILENAME = "batch_results.jsonl"
FINAL_DATASET_FILENAME = "final_classified_captions.csv"
HF_HUB_REPO_ID = "qingy2024/WebVid-100k-ReCaption"

def parse_llm_response(content: str) -> dict:
    """Parses the structured response from the LLM."""
    try:
        # Use regex to find content within the <response> tag
        response_match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        if not response_match:
            return {"classification": "parse_error", "rewritten_caption": "no_response_tag"}

        response_text = response_match.group(1).strip()
        
        # Extract classification and rewritten_caption
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

def process_results(output_file_id, client):
    """Downloads, parses, and prepares the final dataset."""
    print(f"Downloading results from file ID: {output_file_id}...")
    output_file_stream = client.files.download(file_id=output_file_id)

    with open(OUTPUT_FILENAME, 'wb') as f:
        f.write(output_file_stream.read())
    print(f"Results saved to '{OUTPUT_FILENAME}'.")

    processed_data = []
    print("Parsing LLM responses...")
    with open(OUTPUT_FILENAME, "r") as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                print(f"  ...parsed {i} responses")

            data = json.loads(line)
            videoid = int(data["custom_id"])
            
            try:
                llm_output = data["response"]["choices"][0]["message"]["content"]
                parsed = parse_llm_response(llm_output)
                processed_data.append({
                    "videoid": videoid,
                    "classification": parsed["classification"],
                    "rewritten_caption": parsed["rewritten_caption"],
                })
            except (KeyError, IndexError) as e:
                 processed_data.append({
                    "videoid": videoid,
                    "classification": "api_error",
                    "rewritten_caption": f"Error in response structure: {e}",
                })

    print(f"Parsing complete. Total records: {len(processed_data)}")
    
    # Create a Hugging Face Dataset
    final_dataset = Dataset.from_list(processed_data)
    
    print(f"\nUploading dataset to Hugging Face Hub at '{HF_HUB_REPO_ID}'...")
    final_dataset.push_to_hub(HF_HUB_REPO_ID, private=True) # Set private=False for a public repo
    print("--- Upload Complete! ---")
    print(f"Visit your dataset at: https://huggingface.co/datasets/{HF_HUB_REPO_ID}")

def main():
    """Main function to check status and process results."""
    load_dotenv()
    
    api_key = os.getenv("MISTRAL_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    if not api_key or not hf_token:
        raise ValueError("MISTRAL_API_KEY or HF_TOKEN not found in .env file")
        
    # IMPORTANT: Paste the Job ID from the first script here
    job_id = input("Please enter the Batch Job ID from the first script: ")
    if not job_id:
        print("Job ID is required.")
        return

    client = Mistral(api_key=api_key)

    while True:
        retrieved_job = client.batch.jobs.get(job_id=job_id)
        status = retrieved_job.status
        
        print(f"[{time.ctime()}] Job '{job_id}' status: {status}")

        if status == "SUCCESS":
            print("Job completed successfully!")
            process_results(retrieved_job.output_file, client)
            break
        elif status in ["FAILED", "CANCELLED", "TIMEOUT_EXCEEDED"]:
            print(f"Job ended with status: {status}. Error: {retrieved_job.error}")
            break
        elif status in ["QUEUED", "RUNNING"]:
            print("Job is still in progress. Checking again in 5 minutes...")
            time.sleep(300) # Wait for 5 minutes before checking again
        else:
            print(f"Unknown status: {status}. Stopping.")
            break

if __name__ == "__main__":
    main()
