import argparse
import logging
import sys
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset, ClassLabel, Value
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a text classification model.")
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="Hugging Face Dataset ID (e.g., qingy2024/text_classification_dataset, imdb).",
    )
    parser.add_argument(
        "--train_split_name",
        type=str,
        default="train",
        help="Name of the training split (e.g., 'train', 'train[:80%%]').",
    )
    parser.add_argument(
        "--eval_split_name",
        type=str,
        default=None,
        help="Name of the evaluation split (e.g., 'validation', 'test', 'train[80%%:]'). If None, train_split_name will be split.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of the column containing the text.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the column containing the target labels.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Pretrained model identifier from Hugging Face Model Hub (e.g., 'distilbert-base-uncased', 'bert-base-uncased'). Should be a smaller model (<50M params).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./text_classification_output",
        help="Directory to save checkpoints and final model.",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="Log every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--eval_test_size",
        type=float,
        default=0.1,
        help="Fraction of training data to use for evaluation if eval_split_name is not provided."
    )
    parser.add_argument(
        "--num_output",
        action="store_true",
        help="Enable regression mode for numerical score prediction. Target column should contain numerical values."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Script arguments: {args}")

    # 1. Load dataset
    logger.info(f"Loading dataset '{args.dataset_id}'...")
    try:
        if args.eval_split_name:
            raw_datasets = load_dataset(
                args.dataset_id,
                split={
                    "train": args.train_split_name,
                    "validation": args.eval_split_name,
                },
            )
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]
        else:
            logger.info(f"No evaluation split provided. Splitting '{args.train_split_name}' with test_size={args.eval_test_size}.")
            full_train_dataset = load_dataset(args.dataset_id, split=args.train_split_name)
            if not isinstance(full_train_dataset, datasets.Dataset):
                logger.error(f"Expected a single Dataset for split '{args.train_split_name}', but got {type(full_train_dataset)}. Ensure your split name points to a single dataset part.")
                sys.exit(1)

            # Shuffle before splitting if it's a single large dataset
            if len(full_train_dataset) > 1000 : # Arbitrary threshold to decide if shuffling is meaningful
                 full_train_dataset = full_train_dataset.shuffle(seed=args.seed)

            split_datasets = full_train_dataset.train_test_split(
                test_size=args.eval_test_size, seed=args.seed
            )
            train_dataset = split_datasets["train"]
            eval_dataset = split_datasets["test"] # train_test_split names it 'test'

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    # 2. Inspect and prepare labels
    if args.num_output:
        logger.info("Regression mode enabled. Preparing numerical targets...")
        label_feature = train_dataset.features[args.target_column]

        # Check if target column contains numerical values
        if isinstance(label_feature, Value) and (label_feature.dtype.startswith("int") or label_feature.dtype.startswith("float")):
            # Verify all values are numerical
            all_targets = train_dataset[args.target_column] + eval_dataset[args.target_column]
            try:
                all_targets_float = [float(x) for x in all_targets]
                min_val = min(all_targets_float)
                max_val = max(all_targets_float)
                logger.info(f"Target range: [{min_val}, {max_val}]")
                num_labels = 1  # For regression
                id2label = None
                label2id = None
            except (ValueError, TypeError):
                logger.error(f"Target column '{args.target_column}' contains non-numerical values. In regression mode, all targets must be numerical.")
                sys.exit(1)
        else:
            logger.error(f"Target column '{args.target_column}' type '{label_feature}' is not supported for regression. Expected numerical values (int or float).")
            sys.exit(1)
    else:
        logger.info("Classification mode. Preparing labels...")
        label_feature = train_dataset.features[args.target_column]
        id2label = None
        label2id = None

        if isinstance(label_feature, ClassLabel):
            num_labels = label_feature.num_classes
            id2label = {i: label_feature.int2str(i) for i in range(num_labels)}
            label2id = {label: i for i, label in id2label.items()}
            logger.info(f"Target column '{args.target_column}' is ClassLabel with {num_labels} classes.")
            logger.info(f"id2label: {id2label}")
        elif isinstance(label_feature, Value) and label_feature.dtype.startswith("int"):
            # Assuming 0-indexed integer labels if not ClassLabel
            labels = train_dataset[args.target_column] + eval_dataset[args.target_column]
            unique_labels = sorted(list(set(labels)))
            if not all(isinstance(lbl, int) for lbl in unique_labels) or min(unique_labels) < 0 :
                logger.error(f"Target column '{args.target_column}' contains non-integer or negative values. Expected 0-indexed integers or ClassLabel or strings.")
                sys.exit(1)
            num_labels = max(unique_labels) + 1
            logger.info(f"Target column '{args.target_column}' contains integers. Inferred {num_labels} classes (0 to {num_labels-1}).")
            # id2label/label2id can be simple mappings if needed for model config
            id2label = {i: str(i) for i in range(num_labels)}
            label2id = {str(i): i for i in range(num_labels)}
        elif isinstance(label_feature, Value) and label_feature.dtype == "string":
            logger.info(f"Target column '{args.target_column}' contains strings. Creating mapping...")
            # Combine labels from both splits to ensure all are captured
            all_labels_str = train_dataset[args.target_column] + eval_dataset[args.target_column]
            unique_labels_str = sorted(list(set(all_labels_str)))
            num_labels = len(unique_labels_str)
            id2label = {i: label for i, label in enumerate(unique_labels_str)}
            label2id = {label: i for i, label in enumerate(unique_labels_str)}
            logger.info(f"Found {num_labels} unique string labels: {unique_labels_str}")
            logger.info(f"id2label: {id2label}")

            def map_labels_to_int(examples):
                examples["label"] = [label2id[lbl] for lbl in examples[args.target_column]]
                return examples
            train_dataset = train_dataset.map(map_labels_to_int, batched=True, remove_columns=[args.target_column])
            eval_dataset = eval_dataset.map(map_labels_to_int, batched=True, remove_columns=[args.target_column])
            # After mapping, the new column is 'label'
            final_target_column = "label"
        else:
            logger.error(
                f"Unsupported label type '{label_feature}'. Must be ClassLabel, integer, or string."
            )
            sys.exit(1)

        if num_labels <= 1:
            logger.error(f"Found {num_labels} classes. Text classification requires at least 2 classes.")
            sys.exit(1)

    # Ensure the target column is named 'labels' for the Trainer if it's not already
    # This is done after potential string-to-int mapping which creates a 'label' column
    current_target_col_name = final_target_column if 'final_target_column' in locals() else args.target_column
    if current_target_col_name != "labels":
        logger.info(f"Renaming target column '{current_target_col_name}' to 'labels' for the Trainer.")
        train_dataset = train_dataset.rename_column(current_target_col_name, "labels")
        eval_dataset = eval_dataset.rename_column(current_target_col_name, "labels")

    # Convert labels to float for regression mode
    if args.num_output:
        def convert_to_float(examples):
            examples["labels"] = [float(x) for x in examples["labels"]]
            return examples
        train_dataset = train_dataset.map(convert_to_float, batched=True)
        eval_dataset = eval_dataset.map(convert_to_float, batched=True)


    # 3. Load Tokenizer and Model
    logger.info(f"Loading tokenizer and model for '{args.model_name_or_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model_config_kwargs = {"num_labels": num_labels}
        if id2label and label2id: # Pass mappings if available
            model_config_kwargs["id2label"] = id2label
            model_config_kwargs["label2id"] = label2id

        if args.num_output:
            # Use classification model with num_labels=1 for regression
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                num_labels=1  # Regression always has 1 output
            )
        else:
            # Use classification model
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                **model_config_kwargs
            )
    except Exception as e:
        logger.error(f"Failed to load tokenizer or model: {e}")
        sys.exit(1)

    # Check model size (parameters)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model '{args.model_name_or_path}' has {num_params:,} trainable parameters.")
    if num_params > 50_000_000:
        logger.warning(
            f"Warning: The model '{args.model_name_or_path}' has {num_params:,} parameters, "
            "which is greater than the suggested <50M."
        )

    # 4. Preprocess (Tokenize)
    logger.info("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_column],
            padding="max_length", # Pad to max_length, can also use "longest" for dynamic padding
            truncation=True,
            max_length=args.max_seq_length,
        )

    try:
        # Remove original text column to avoid issues, keep only tokenized fields and labels
        # If the text_column is also the target_column (unlikely but possible), don't remove it here.
        remove_cols = [args.text_column] if args.text_column != "labels" else []

        tokenized_train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_cols + ([args.target_column] if args.target_column in train_dataset.column_names and args.target_column != "labels" else [])
        )
        tokenized_eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_cols + ([args.target_column] if args.target_column in eval_dataset.column_names and args.target_column != "labels" else [])
        )
    except KeyError as e:
        logger.error(f"KeyError during tokenization or column removal: {e}. "
                     f"Ensure '{args.text_column}' and (if applicable) '{args.target_column}' exist and are correctly specified.")
        logger.error(f"Train dataset columns: {train_dataset.column_names}")
        logger.error(f"Eval dataset columns: {eval_dataset.column_names}")
        sys.exit(1)


    # Set format for PyTorch
    tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


    # 5. Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Compute Metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if args.num_output:
            # Regression metrics
            predictions = predictions.flatten()  # Remove extra dimensions
            mse = mean_squared_error(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            # Calculate R² manually
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            return {"mse": mse, "mae": mae, "r2": r2}
        else:
            # Classification metrics
            predictions = np.argmax(predictions, axis=1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="weighted") # Use "macro" or "micro" if preferred
            return {"accuracy": acc, "f1": f1}

    # 7. Training Arguments
    logger.info("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01, # Standard for AdamW
        eval_strategy="steps", # Evaluate at each logging/saving step
        eval_steps=args.log_freq,    # Evaluate every log_freq steps
        logging_steps=args.log_freq,
        save_steps=args.save_steps,
        save_total_limit=3, # Keep only the last 3 checkpoints + the best one
        load_best_model_at_end=True, # Load the best model found during training at the end
        metric_for_best_model="r2" if args.num_output else "f1", # or "accuracy"
        greater_is_better=True,
        push_to_hub=False, # Set to True if you want to push to Hugging Face Hub
        report_to="tensorboard", # Can also use "wandb"
        seed=args.seed,
        fp16=torch.cuda.is_available(), # Use mixed precision if CUDA is available
    )

    # 8. Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer, # Good to pass for auto-handling of padding and saving
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9. Train
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("Training finished.")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() # Saves optimizer, scheduler, etc.

        # Save the fine-tuned model and tokenizer
        logger.info(f"Saving best model to {args.output_dir}...")
        trainer.save_model(args.output_dir) # Saves the best model because load_best_model_at_end=True
        # tokenizer.save_pretrained(args.output_dir) # Trainer also saves tokenizer if passed

        logger.info("Final evaluation on the evaluation set:")
        eval_metrics = trainer.evaluate(eval_dataset=tokenized_eval_dataset)
        trainer.log_metrics("eval_final", eval_metrics)
        trainer.save_metrics("eval_final", eval_metrics)
        logger.info(f"Final Eval metrics: {eval_metrics}")

    except Exception as e:
        logger.error(f"An error occurred during training or evaluation: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Script finished. Checkpoints and model saved in {args.output_dir}")

if __name__ == "__main__":
    main()

"""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust cuXXX for your CUDA version, or use cpuonly
pip install transformers datasets scikit-learn tensorboard accelerate # accelerate is for fp16 and multi-GPU

# Classification example:
python text_classifier_ft.py \
    --dataset_id imdb \
    --train_split_name "train[:2000]" \
    --eval_split_name "test[:500]" \
    --text_column text \
    --target_column label \
    --model_name_or_path distilbert/distilbert-base-uncased \
    --output_dir ./text_classifier_output \
    --log_freq 50 \
    --save_steps 100 \
    --num_epochs 2 \
    --batch_size 16 \
    --max_seq_length 256

# Regression example (for numerical score prediction):
python text_classifier_ft.py \
    --dataset_id your_dataset_with_scores \
    --train_split_name "train" \
    --text_column text \
    --target_column score \
    --model_name_or_path distilbert/distilbert-base-uncased \
    --output_dir ./score_regression_output \
    --num_output \
    --num_epochs 3 \
    --batch_size 16 \
    --max_seq_length 256
"""
