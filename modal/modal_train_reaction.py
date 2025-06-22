import os
import secrets
from random import randint
import subprocess
import pathlib
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .pip_install("jupyterlab")
    .pip_install("ipywidgets")
    .pip_install("hf_transfer")
    .pip_install("jupyter")
    .pip_install('packaging')
    .pip_install('ninja')
    .run_commands(
        "apt-get update -y",
        "apt-get install git curl -y",
    )
    .pip_install('torch>=2.4.0')
    .pip_install('torchvision')
    .pip_install('numpy')
    .pip_install('wandb')
    .pip_install('pandas')
    .pip_install('tensorboard')
    .run_commands('pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.7.4.post1+cu126torch2.4-cp310-cp310-linux_x86_64.whl')
    .pip_install('flash-attn')
    .pip_install('vllm==0.7.3')
    .pip_install(
        "bitsandbytes==0.45.3",
        "protobuf==6.31.0",
    )
    .pip_install('pyzmq==26.4.0')
    .pip_install('accelerate==1.7.0')
    .pip_install('xformers==0.0.29.post3')
    .pip_install('peft==0.15.2')
    .pip_install('triton==2.3.0')
    .pip_install('cut_cross_entropy==25.1.1')
    .pip_install('unsloth_zoo')
    .pip_install('sentencepiece==0.2.0')
    .pip_install('datasets==3.6.0')
    .pip_install('huggingface-hub==0.31.2')
    .pip_install('unsloth')
    .pip_install('evaluate==0.4.3')
    .pip_install('regex==2024.11.6')
    .pip_install('matplotlib==3.10.3')
)

image = image.env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_TOKEN": os.environ['HF_TOKEN'],
    "WANDB_API_KEY": os.environ['WANDB_API_KEY'],
    "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"
})

image = image.run_commands(
    "huggingface-cli download unsloth/Qwen2.5-1.5B-Instruct",
)

app = modal.App(image=image, name="ReAction Training")

@app.function(gpu="H100:1", timeout=86400)
def runwithgpu():
    from unsloth import FastLanguageModel
    from datasets import load_dataset, Dataset
    import os
    import torch
    import random
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting model training script")

    max_seq_length = 16384
    dtype = None

    logger.info(f"Loading model with max_seq_length={max_seq_length}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning = True,
    )
    logger.info("Model loaded successfully!")

    logger.info(f"Chat template is:\n{tokenizer.chat_template}\n")

    EOS_TOKEN = tokenizer.eos_token

    logger.info(f"Using EOS Token: {EOS_TOKEN}")
    logger.info(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    logger.info("Chat Template set up!")

    # Testing chat template
    template_test = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Action"}, {"role": "assistant", "content": "Rewritten Action"}, {"role": "user", "content": "another action"}],
        tokenize = False,
        add_generation_prompt = True,
    )
    logger.info(f"Chat template test:\n\n{template_test}\n\n")

    BASE_SEED = 42
    logger.info(f"Using BASE_SEED={BASE_SEED}")

    random.seed(BASE_SEED)

    def formatting_prompts_func(examples):
        original_captions = examples["original_caption"]
        rewritten_captions = examples["rewritten_caption"]

        texts = []

        for i in range(len(original_captions)):
            original_caption = original_captions[i]
            rewritten_caption = rewritten_captions[i]

            # Create the chat-template-formatted string
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are ReAction, an assistant to rewrite video caption to make them clearer. Rewrite the user's video caption."} ,{"role": "user", "content": original_caption}, {"role": "assistant", "content": rewritten_caption}],
                tokenize = False,
                add_generation_prompt = False,
            ).strip()
            texts.append(text)

        return {"text": texts}

    logger.info("Loading dataset...")
    dataset = load_dataset("qingy2024/webvid-10M-classified", split = "train")

    dataset = dataset.filter(lambda row: row['classification'] == 'action')

    dataset = dataset.map(formatting_prompts_func, batched = True,)
    logger.info(f"Dataset loaded with {len(dataset)} examples")

    # Sample a random example to check formatting
    index = random.randint(0, len(dataset) - 1)
    logger.info(f"==== Sample example (index {index}) ====")
    logger.info(dataset[index]['text'])
    inputs = tokenizer(dataset[index]['text'], return_tensors="pt").to(model.device)
    logger.info("Tokenized sample:")
    logger.info(inputs)

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 2,
            warmup_steps = 180,
            num_train_epochs = 1,
            learning_rate = 5e-5,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            save_strategy = "no",
            seed = 3407,
            output_dir = "outputs",
            run_name="ReAction-1.5B",
            report_to = "wandb",
        ),
    )
    logger.info("SFTTrainer setup complete")

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.info("Training completed")

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    logger.info("Pushing model to Hugging Face Hub...")
    model.push_to_hub("qingy2024/ReAction-1.5B", token = os.environ['HF_TOKEN'])
    tokenizer.push_to_hub("qingy2024/ReAction-1.5B", token = os.environ['HF_TOKEN'])
    logger.info("Model and tokenizer pushed to hub successfully")

    logger.info(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Print token IDs for verification
    logger.info(f"ID for {EOS_TOKEN}: {tokenizer.eos_token_id}")
    logger.info(f"ID for PAD: {tokenizer.pad_token_id}")

    logger.info("Script completed successfully")

# Define a local entrypoint function for the Modal application.
@app.local_entrypoint()
def main():
    print("========== Running with GPU!! ==========")
    runwithgpu.remote()
