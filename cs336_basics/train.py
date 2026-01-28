import argparse
import math
import os
import time

import numpy as np
import torch
import wandb

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.utils import load_checkpoint, save_checkpoint, save_config


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized data (numpy array)")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--wandb_project", type=str, default="project_x")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    args.device = get_default_device()
    print(f"Using device: {args.device}")

    if args.wandb_run_name is None:
        args.wandb_run_name = (
            f"ts-d{args.d_model}-l{args.num_layers}-bs{args.batch_size}-h{args.num_heads}-lr{args.lr:.1e}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        "vocab_size": args.vocab_size,
        "max_seq_len": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    save_config(config, os.path.join(args.output_dir, "config.json"))
    print(f"Saved model config to {os.path.join(args.output_dir, 'config.json')}")

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    dataset = np.memmap(args.data_path, dtype=np.uint16, mode="r")
    print(f"Dataset size: {len(dataset)} tokens")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iter = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            start_iter = load_checkpoint(args.resume, model, optimizer)
        else:
            print(f"Checkpoint not found at {args.resume}, starting from scratch.")

    # Training loop
    model.train()
    start_time = time.time()

    for it in range(start_iter, args.max_iters):
        # 1. Get Batch
        x, y = get_batch(dataset, args.batch_size, args.context_length, args.device)

        # 2. LR Schedule
        lr = get_lr_cosine_schedule(it, args.lr, args.lr * 0.1, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 3. Forward Pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()

        # 5. Gradient Clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # 6. Optimizer step
        optimizer.step()

        # Logging
        if it % args.log_interval == 0:
            dt = time.time() - start_time
            tps = args.batch_size * args.context_length * args.log_interval / dt if it > start_iter else 0
            print(f"Iter {it}: Loss {loss.item():.4f}, LR {lr:.2e}, Time {dt:.2f}s, Tokens/s {tps:.0f}")

            if not args.no_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/tokens_per_sec": tps, "train/step": it})

            start_time = time.time()

        if it > 0 and it % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, save_path)
            print(f"Saved checkpoint to {save_path}")

    final_path = os.path.join(args.output_dir, "final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Training complete. Saved final model to {final_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
