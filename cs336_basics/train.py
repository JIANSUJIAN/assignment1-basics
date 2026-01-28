import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.logging import ExperimentLogger
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.utils import load_checkpoint, save_checkpoint


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    val_dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 10,
) -> float:
    """
    Evaluate model on validation dataset.

    Args:
        model: The model to evaluate
        val_dataset: Validation dataset (memory-mapped numpy array)
        batch_size: Batch size for evaluation
        context_length: Context length
        device: Device to use
        num_batches: Number of batches to evaluate

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = get_batch(val_dataset, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        total_loss += loss.item()

    model.train()
    return total_loss / num_batches


def train():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized training data (numpy array)")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to tokenized validation data (numpy array)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints and logs")

    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR as ratio of peak LR")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_iters", type=int, default=5000, help="Maximum training iterations")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Learning rate warmup iterations")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging")
    parser.add_argument("--val_interval", type=int, default=500, help="Steps between validation evaluations")
    parser.add_argument("--val_batches", type=int, default=20, help="Number of batches for validation evaluation")
    parser.add_argument("--save_interval", type=int, default=1000, help="Steps between checkpoint saves")

    # Experiment tracking arguments
    parser.add_argument("--experiment_name", type=str, default=None, help="Name for this experiment")
    parser.add_argument("--wandb_project", type=str, default="cs336", help="Weights & Biases project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")

    # Checkpoint arguments
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Set device
    args.device = get_default_device()
    print(f"Using device: {args.device}")

    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = (
            f"d{args.d_model}-l{args.num_layers}-h{args.num_heads}-bs{args.batch_size}-lr{args.lr:.1e}"
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build config dictionaries
    model_config = {
        "vocab_size": args.vocab_size,
        "max_seq_len": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }

    training_config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.weight_decay,
        "max_iters": args.max_iters,
        "warmup_iters": args.warmup_iters,
        "grad_clip": args.grad_clip,
        "context_length": args.context_length,
    }

    full_config = {
        "model": model_config,
        "training": training_config,
        "data_path": args.data_path,
        "val_data_path": args.val_data_path,
        "device": args.device,
    }

    # Initialize experiment logger
    logger = ExperimentLogger(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        config=full_config,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.experiment_name,
    )

    # Load training data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Training data file not found: {args.data_path}")
    train_dataset = np.memmap(args.data_path, dtype=np.uint16, mode="r")
    print(f"Training dataset size: {len(train_dataset):,} tokens")

    # Load validation data if provided
    val_dataset = None
    if args.val_data_path:
        if not os.path.exists(args.val_data_path):
            raise FileNotFoundError(f"Validation data file not found: {args.val_data_path}")
        val_dataset = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")
        print(f"Validation dataset size: {len(val_dataset):,} tokens")

    # Initialize model
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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume from checkpoint if provided
    start_iter = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            start_iter = load_checkpoint(args.resume, model, optimizer)
        else:
            print(f"Checkpoint not found at {args.resume}, starting from scratch.")

    # Training loop
    model.train()
    tokens_since_last_log = 0
    last_log_time = time.time()

    for it in range(start_iter, args.max_iters):
        # 1. Get Batch
        x, y = get_batch(train_dataset, args.batch_size, args.context_length, args.device)

        # 2. LR Schedule
        min_lr = args.lr * args.min_lr_ratio
        lr = get_lr_cosine_schedule(it, args.lr, min_lr, args.warmup_iters, args.max_iters)
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

        # Track tokens processed
        tokens_since_last_log += args.batch_size * args.context_length

        # Logging
        if it % args.log_interval == 0:
            current_time = time.time()
            dt = current_time - last_log_time
            tokens_per_sec = tokens_since_last_log / dt if dt > 0 and it > start_iter else 0

            logger.log_step(
                step=it,
                train_loss=loss.item(),
                learning_rate=lr,
                tokens_per_sec=tokens_per_sec,
                device=args.device,
            )

            # Reset counters
            tokens_since_last_log = 0
            last_log_time = current_time

        # Validation evaluation
        if val_dataset is not None and it > 0 and it % args.val_interval == 0:
            val_loss = evaluate_validation(
                model=model,
                val_dataset=val_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.val_batches,
            )
            logger.log_validation(step=it, val_loss=val_loss)

        # Checkpoint saving
        if it > 0 and it % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, save_path)
            print(f"Saved checkpoint to {save_path}")

    # Final validation evaluation
    if val_dataset is not None:
        final_val_loss = evaluate_validation(
            model=model,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            num_batches=args.val_batches,
        )
        logger.log_validation(step=args.max_iters, val_loss=final_val_loss)

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Saved final model to {final_path}")

    # Finalize logging
    logger.finish(total_steps=args.max_iters)


if __name__ == "__main__":
    train()
