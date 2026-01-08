#!/usr/bin/env python3
"""
Helper script for fine-tuning Spanish TTS models.

This script wraps run_vits_finetuning.py and simplifies the process of fine-tuning
the Spanish MMS-TTS checkpoint (ylacombe/mms-tts-spa-train) with custom datasets.

Usage:
    python finetune_spanish.py \
      --train_dataset path/to/common_voice_es_train.jsonl \
      --eval_dataset  path/to/common_voice_es_val.jsonl \
      --output_dir    path/to/output_dir \
      --num_train_steps 1500 \
      --per_device_train_batch_size 16 \
      --learning_rate 5e-5
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Spanish TTS model using MMS-TTS-SPA checkpoint"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Path to training dataset file (JSONL or CSV format)"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset file (JSONL or CSV format)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where model checkpoints and logs will be saved"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1500,
        help="Total number of training steps (default: 1500)"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device for training (default: 16)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ylacombe/mms-tts-spa-train",
        help="Model checkpoint to use (default: ylacombe/mms-tts-spa-train)"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID to push to Hugging Face Hub (optional)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to Hugging Face Hub"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.train_dataset):
        print(f"Error: Training dataset not found: {args.train_dataset}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.eval_dataset):
        print(f"Error: Evaluation dataset not found: {args.eval_dataset}", file=sys.stderr)
        sys.exit(1)
    
    # Determine dataset format
    train_ext = os.path.splitext(args.train_dataset)[1].lower()
    eval_ext = os.path.splitext(args.eval_dataset)[1].lower()
    
    if train_ext not in ['.jsonl', '.json', '.csv']:
        print(f"Warning: Training dataset extension '{train_ext}' not recognized. Supported: .jsonl, .json, .csv", file=sys.stderr)
    if eval_ext not in ['.jsonl', '.json', '.csv']:
        print(f"Warning: Evaluation dataset extension '{eval_ext}' not recognized. Supported: .jsonl, .json, .csv", file=sys.stderr)
    
    # Build command to run accelerate launch with run_vits_finetuning.py
    cmd = [
        "accelerate", "launch",
        "run_vits_finetuning.py",
        "--model_name_or_path", args.model_name_or_path,
        "--output_dir", args.output_dir,
        "--learning_rate", str(args.learning_rate),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--num_train_steps", str(args.num_train_steps),
        "--do_train",
        "--do_eval",
        "--train_dataset_file", args.train_dataset,
        "--eval_dataset_file", args.eval_dataset,
    ]
    
    # Add optional arguments
    if args.hub_model_id:
        cmd.extend(["--hub_model_id", args.hub_model_id])
    
    if args.push_to_hub:
        cmd.append("--push_to_hub")
    
    if args.use_wandb:
        cmd.extend(["--report_to", "wandb"])
    
    # Print command for debugging
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Training script failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
