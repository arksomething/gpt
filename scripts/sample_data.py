#!/usr/bin/env python3
"""
Sample random sequences from training data to inspect what the model sees.

Features:
- Random or sequential sampling
- Save samples to file
- Interactive mode for browsing
- Screen/tmux auto-launch for batch operations
- File logging
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import sentencepiece as spm
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Tee:
    """Duplicate stdout to a log file."""

    def __init__(self, log_path, stream_name="stdout"):
        self.log_path = log_path
        self.stream_name = stream_name
        self.original = getattr(sys, stream_name)
        self.log_file = open(log_path, "a", buffering=1, encoding="utf-8")
        setattr(sys, stream_name, self)

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def close(self):
        setattr(sys, self.stream_name, self.original)
        self.log_file.close()


def maybe_launch_screen(enabled: bool, session_name: Optional[str] = None) -> bool:
    """Launch in screen/tmux if requested and not already in one."""
    if not enabled:
        return False
    if os.environ.get("STY") or os.environ.get("TMUX"):
        return False  # Already in screen/tmux

    session = session_name or f"sample-{time.strftime('%Y%m%d-%H%M%S')}"
    command = [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:]]
    # Remove --launch_screen to avoid infinite recursion
    command = [c for c in command if c != "--launch_screen"]

    if shutil.which("screen") is not None:
        subprocess.check_call(["screen", "-dmS", session, *command])
        print(f"Started screen session '{session}'. Attach with: screen -r {session}")
        return True

    if shutil.which("tmux") is not None:
        subprocess.check_call(["tmux", "new-session", "-d", "-s", session, *command])
        print(f"Started tmux session '{session}'. Attach with: tmux attach -t {session}")
        return True

    print("screen/tmux not found; running in foreground.")
    return False


def format_sample(
    sample_num: int,
    total_samples: int,
    offset: int,
    total_tokens: int,
    text: str,
    num_docs: int,
    tokens: Optional[List[int]] = None,
    eos_positions: Optional[List[int]] = None,
    show_raw: bool = False,
) -> str:
    """Format a single sample for display."""
    lines = []
    lines.append(f"[Sample {sample_num}/{total_samples}] offset={offset:,} ({offset/total_tokens*100:.2f}%)")
    lines.append(f"Documents in sample: {num_docs}")
    lines.append("-" * 80)
    
    if show_raw and tokens is not None:
        lines.append(f"Token IDs (first 50): {tokens[:50]}")
        if eos_positions is not None:
            lines.append(f"EOS positions: {eos_positions}")
        lines.append("")
    
    lines.append(text)
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    return "\n".join(lines)


def decode_with_eos_markers(
    tokens: List[int],
    sp: spm.SentencePieceProcessor,
    eos_token_id: int = 2,
) -> tuple:
    """Decode tokens, making EOS boundaries visible. Returns (text, num_docs, eos_positions)."""
    eos_positions = [j for j, t in enumerate(tokens) if t == eos_token_id]
    
    if eos_positions:
        parts = []
        prev_pos = 0
        for eos_pos in eos_positions:
            if eos_pos > prev_pos:
                part_tokens = tokens[prev_pos:eos_pos]
                parts.append(sp.decode(part_tokens))
            prev_pos = eos_pos + 1
        if prev_pos < len(tokens):
            parts.append(sp.decode(tokens[prev_pos:]))
        text = "\n[EOS]\n".join(parts)
    else:
        text = sp.decode(tokens)
    
    num_docs = len(eos_positions) + 1
    return text, num_docs, eos_positions


def sample_data(
    data: np.ndarray,
    sp: spm.SentencePieceProcessor,
    num_samples: int,
    block_size: int,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
    show_raw: bool = False,
    eos_token_id: int = 2,
) -> List[str]:
    """Generate formatted samples from data."""
    total_tokens = len(data)
    max_start = total_tokens - block_size
    
    if max_start <= 0:
        raise ValueError("Data file too small for the requested block_size")
    
    rng = np.random.default_rng(seed)
    samples = []
    
    for i in range(num_samples):
        if offset is not None:
            start = offset + i * block_size
            if start >= max_start:
                break
        else:
            start = rng.integers(0, max_start)
        
        tokens = data[start : start + block_size].tolist()
        text, num_docs, eos_positions = decode_with_eos_markers(tokens, sp, eos_token_id)
        
        sample_str = format_sample(
            sample_num=i + 1,
            total_samples=num_samples,
            offset=start,
            total_tokens=total_tokens,
            text=text,
            num_docs=num_docs,
            tokens=tokens if show_raw else None,
            eos_positions=eos_positions if show_raw else None,
            show_raw=show_raw,
        )
        samples.append(sample_str)
    
    return samples


def interactive_mode(
    data: np.ndarray,
    sp: spm.SentencePieceProcessor,
    block_size: int,
    eos_token_id: int = 2,
):
    """Interactive browsing mode."""
    total_tokens = len(data)
    max_start = total_tokens - block_size
    rng = np.random.default_rng()
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Commands:")
    print("  [Enter]     - Random sample")
    print("  [number]    - Sample at specific offset")
    print("  r           - Random sample (same as Enter)")
    print("  n           - Next sequential sample")
    print("  p           - Previous sequential sample")
    print("  s [file]    - Save current sample to file")
    print("  q           - Quit")
    print("=" * 80)
    print()
    
    current_offset = rng.integers(0, max_start)
    current_sample = None
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break
        
        if cmd == "q":
            print("Exiting interactive mode.")
            break
        elif cmd == "" or cmd == "r":
            current_offset = rng.integers(0, max_start)
        elif cmd == "n":
            current_offset = min(current_offset + block_size, max_start - 1)
        elif cmd == "p":
            current_offset = max(current_offset - block_size, 0)
        elif cmd.startswith("s"):
            parts = cmd.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else f"sample_{current_offset}.txt"
            if current_sample:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(current_sample)
                print(f"Saved to {filename}")
            else:
                print("No sample to save yet.")
            continue
        elif cmd.isdigit():
            current_offset = min(int(cmd), max_start - 1)
        else:
            print("Unknown command. Type 'q' to quit.")
            continue
        
        # Generate and display sample
        tokens = data[current_offset : current_offset + block_size].tolist()
        text, num_docs, eos_positions = decode_with_eos_markers(tokens, sp, eos_token_id)
        
        current_sample = format_sample(
            sample_num=1,
            total_samples=1,
            offset=current_offset,
            total_tokens=total_tokens,
            text=text,
            num_docs=num_docs,
        )
        print(current_sample)


def main():
    parser = argparse.ArgumentParser(
        description="Sample random sequences from training data and decode them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sampling
  %(prog)s -n 10
  
  # Save samples to file
  %(prog)s -n 20 --output samples.txt
  
  # Interactive browsing
  %(prog)s --interactive
  
  # Run in background with screen
  %(prog)s -n 100 --output samples.txt --launch_screen
  
  # Sample from specific offset
  %(prog)s --offset 1000000 -n 5
        """,
    )
    
    # Input options
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--data", default=None, help="Path to .bin data file (overrides config)")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer model (overrides config)")
    parser.add_argument("--val", action="store_true", help="Sample from validation data instead of training")
    
    # Sampling options
    parser.add_argument("-n", "--num_samples", type=int, default=10, help="Number of samples to take")
    parser.add_argument("--block_size", type=int, default=None, help="Sequence length (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--offset", type=int, default=None, help="Sample from specific offset instead of random")
    parser.add_argument("--raw", action="store_true", help="Also show raw token IDs")
    
    # Output options
    parser.add_argument("-o", "--output", default=None, help="Save samples to file")
    parser.add_argument("--format", choices=["text", "json", "jsonl"], default="text", help="Output format")
    parser.add_argument("--quiet", action="store_true", help="Don't print to console (only write to file)")
    
    # Mode options
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive browsing mode")
    parser.add_argument("--launch_screen", action="store_true", help="Run in screen/tmux session")
    parser.add_argument("--screen_name", default=None, help="Name for screen/tmux session")
    parser.add_argument("--log_file", default=None, help="Log output to file (in addition to console)")
    
    args = parser.parse_args()
    
    # Maybe launch in screen/tmux
    if maybe_launch_screen(args.launch_screen, args.screen_name):
        return
    
    # Set up logging
    log_cleanup = lambda: None
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        tee = Tee(args.log_file, "stdout")
        log_cleanup = tee.close
    
    try:
        # Load config
        cfg = load_yaml(args.config)
        data_cfg = cfg["data"]
        
        # Resolve paths
        data_path = args.data
        if data_path is None:
            data_path = data_cfg["val_bin"] if args.val else data_cfg["train_bin"]
        
        tokenizer_path = args.tokenizer
        if tokenizer_path is None:
            tokenizer_path = cfg.get("data_prep", {}).get("tokenizer_model", "tokenizer/spm.model")
        
        block_size = args.block_size or data_cfg.get("block_size", 2048)
        dtype = np.dtype(data_cfg.get("dtype", "uint16"))
        
        # Load data
        if not os.path.exists(data_path):
            raise SystemExit(f"Data file not found: {data_path}")
        data = np.memmap(data_path, dtype=dtype, mode="r")
        total_tokens = len(data)
        
        if not args.quiet:
            print(f"Loaded {total_tokens:,} tokens from {data_path}")
            print(f"Block size: {block_size}")
            print()
        
        # Load tokenizer
        if not os.path.exists(tokenizer_path):
            raise SystemExit(f"Tokenizer not found: {tokenizer_path}")
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        
        if not args.quiet:
            print(f"Loaded tokenizer: {tokenizer_path} (vocab_size={sp.vocab_size()})")
            print("=" * 80)
            print()
        
        # Interactive mode
        if args.interactive:
            interactive_mode(data, sp, block_size)
            return
        
        # Generate samples
        samples = sample_data(
            data=data,
            sp=sp,
            num_samples=args.num_samples,
            block_size=block_size,
            seed=args.seed,
            offset=args.offset,
            show_raw=args.raw,
        )
        
        # Output
        if args.format == "text":
            output_text = "\n".join(samples)
        elif args.format == "json":
            import json
            output_text = json.dumps({"samples": samples, "timestamp": datetime.utcnow().isoformat()}, indent=2)
        elif args.format == "jsonl":
            import json
            output_text = "\n".join(json.dumps({"sample": s}) for s in samples)
        
        # Write to file
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_text)
            if not args.quiet:
                print(f"\nSaved {len(samples)} samples to {args.output}")
        
        # Print to console
        if not args.quiet:
            if args.format == "text":
                for sample in samples:
                    print(sample)
            else:
                print(output_text)
    
    finally:
        log_cleanup()


if __name__ == "__main__":
    main()
