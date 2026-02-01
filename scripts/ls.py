#!/usr/bin/env python3
"""List all available commands with their flags, defaults, and descriptions."""

COMMANDS = [
    {
        "name": "train",
        "description": "Train a ~100M parameter Llama-style model.",
        "args": [
            ("--model_config", "configs/model_100m.yaml", "Model architecture config file"),
            ("--train_config", "configs/train.yaml", "Training hyperparameters config file"),
            ("--resume_from", "None", "Path to checkpoint directory to resume from"),
            ("--resume_from_slot", "None", "Resume from named slot (last, best, good_1, etc.)"),
            ("--smoke", "False", "Quick smoke test run (max 50 steps)"),
            ("--launch_screen", "False", "Auto-launch in screen/tmux session"),
            ("--screen_name", "None", "Custom session name for screen/tmux"),
        ],
    },
    {
        "name": "eval",
        "description": "Evaluate a checkpoint and generate samples.",
        "args": [
            ("--model_config", "configs/model_100m.yaml", "Model architecture config file"),
            ("--train_config", "configs/train.yaml", "Training config (for eval defaults)"),
            ("--checkpoint", "runs/llama-100m/final", "Path to checkpoint directory"),
            ("--tokenizer_model", "tokenizer/spm.model", "SentencePiece model file"),
            ("--batches", "200", "Number of eval batches for loss calculation"),
            ("--prompt", "The quick brown fox", "Text prompt for generation"),
            ("--min_new_tokens", "16", "Minimum tokens to generate"),
            ("--max_new_tokens", "100", "Maximum tokens to generate"),
            ("--temperature", "0.7", "Sampling temperature"),
            ("--top_p", "0.9", "Nucleus sampling probability"),
            ("--top_k", "50", "Top-k sampling cutoff"),
            ("--repetition_penalty", "1.1", "Repetition penalty factor"),
        ],
    },
    {
        "name": "prepare-data",
        "description": "Prepare memmapped token data from C4 + Wikipedia + Gutenberg (filtered).",
        "args": [
            ("--config", "configs/train.yaml", "Config file with data_prep section"),
            ("--tokenizer_model", "tokenizer/spm.model", "SentencePiece model file"),
            ("--out_dir", "data", "Output directory for .bin files"),
            ("--train_tokens", "2_000_000_000", "Target number of training tokens"),
            ("--val_tokens", "20_000_000", "Target number of validation tokens"),
            ("--c4_weight", "0.30", "C4 dataset sampling weight"),
            ("--wiki_weight", "0.45", "Wikipedia dataset sampling weight"),
            ("--gutenberg_weight", "0.25", "Gutenberg dataset sampling weight"),
            ("--shuffle_buffer", "50_000", "Streaming shuffle buffer size"),
            ("--seed", "1337", "Random seed"),
            ("--log_interval", "30", "Seconds between progress logs"),
            ("--overwrite", "False", "Overwrite existing .bin files"),
        ],
    },
    {
        "name": "train-tokenizer",
        "description": "Train a SentencePiece BPE tokenizer from C4 + Wikipedia + Gutenberg.",
        "args": [
            ("--config", "configs/train.yaml", "Config file with tokenizer_training section"),
            ("--output_dir", "tokenizer", "Output directory for tokenizer files"),
            ("--vocab_size", "32000", "Vocabulary size"),
            ("--c4_weight", "0.30", "C4 dataset sampling weight"),
            ("--wiki_weight", "0.45", "Wikipedia dataset sampling weight"),
            ("--gutenberg_weight", "0.25", "Gutenberg dataset sampling weight"),
            ("--min_chars", "200", "Minimum characters per document"),
            ("--max_chars", "50_000_000", "Maximum characters to use for training"),
            ("--shuffle_buffer", "50_000", "Streaming shuffle buffer size"),
            ("--seed", "1337", "Random seed"),
        ],
    },
    {
        "name": "benchmark",
        "description": "Benchmark training throughput (tokens/sec).",
        "args": [
            ("--model_config", "configs/model_100m.yaml", "Model architecture config file"),
            ("--train_config", "configs/train.yaml", "Training config file"),
            ("--warmup_steps", "10", "Warmup steps before timing"),
            ("--steps", "50", "Steps to measure"),
            ("--output_path", "runs/throughput.json", "Output JSON file path"),
        ],
    },
    {
        "name": "tokenizer-repl",
        "description": "Inspect tokenizer and sample from tokenized data.",
        "args": [
            ("--tokenizer_model", "tokenizer/spm.model", "SentencePiece model file"),
            ("--train_bin", "data/train.bin", "Training data .bin file"),
            ("--val_bin", "data/val.bin", "Validation data .bin file"),
            ("--dtype", "uint16", "Data type of .bin files"),
            ("--seed", "1337", "Random seed for sampling"),
            ("--encode", "None", "Text to encode (one-shot mode)"),
            ("--decode", "None", "Token IDs to decode (one-shot mode)"),
            ("--sample", "False", "Sample random sequences from data"),
            ("--split", "train", "Which split to sample from (train/val)"),
            ("--length", "128", "Sample sequence length"),
            ("--count", "3", "Number of samples"),
            ("--max_show_tokens", "120", "Max tokens to display"),
            ("--max_text_chars", "800", "Max decoded text chars to display"),
            ("--repl", "False", "Start interactive REPL (default if no flags)"),
        ],
    },
    {
        "name": "scan-bins",
        "description": "Scan tokenized .bin files for range and frequency issues.",
        "args": [
            ("--train", "None", "Path to train.bin"),
            ("--val", "None", "Path to val.bin"),
            ("--vocab-size", "32000", "Expected vocabulary size"),
            ("--dtype", "uint16", "Data type of .bin files"),
            ("--topk", "20", "Number of top tokens to show"),
            ("--special-ids", "[0, 1, 2, 3]", "Token IDs to report frequency for"),
            ("--chunk-size", "1_000_000", "Tokens per chunk when scanning"),
        ],
    },
]


def main():
    print("Available commands (run with `uv run <command>`):\n")
    print("=" * 72)

    for cmd in COMMANDS:
        print(f"\n  {cmd['name']}")
        print(f"  {'-' * len(cmd['name'])}")
        print(f"  {cmd['description']}\n")

        if cmd["args"]:
            # Calculate column widths
            flag_width = max(len(arg[0]) for arg in cmd["args"])
            default_width = max(len(str(arg[1])) for arg in cmd["args"])

            print(f"  {'Flag':<{flag_width}}  {'Default':<{default_width}}  Description")
            print(f"  {'-' * flag_width}  {'-' * default_width}  {'-' * 30}")

            for flag, default, desc in cmd["args"]:
                print(f"  {flag:<{flag_width}}  {str(default):<{default_width}}  {desc}")

    print("\n" + "=" * 72)
    print("\nExamples:")
    print("  uv run train --smoke                    # Quick test run")
    print("  uv run train --launch_screen            # Train in background")
    print("  uv run eval --checkpoint runs/llama-100m/step_0010000")
    print("  uv run prepare-data --train_tokens 1_000_000_000")
    print("  uv run benchmark --steps 100")
    print("  uv run tokenizer-repl --sample --count 5")
    print("  uv run scan-bins --train data/train.bin --val data/val.bin")
    print()


if __name__ == "__main__":
    main()
