"""
Experiment logging infrastructure for tracking training runs.

Supports:
- Logging metrics with respect to gradient steps and wallclock time
- Both wandb and local JSON logging
- Training and validation loss tracking
- Gradient norms, parameter norms, and memory usage
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import psutil
import torch


@dataclass
class ExperimentMetrics:
    """Container for metrics at a single logging step."""

    step: int
    wallclock_time: float  # seconds since training start
    train_loss: float | None = None
    val_loss: float | None = None
    learning_rate: float | None = None
    tokens_per_sec: float | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    process_memory_mb: float | None = None


@dataclass
class ExperimentSummary:
    """Summary of a complete experiment run."""

    experiment_name: str
    start_time: str
    end_time: str | None = None
    total_steps: int = 0
    total_wallclock_seconds: float = 0.0
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_val_loss: float | None = None
    best_val_step: int | None = None
    config: dict = field(default_factory=dict)
    notes: str = ""


class ExperimentLogger:
    """
    Unified experiment logging for training runs.

    Logs to both wandb (if enabled) and local JSON files.
    Tracks wallclock time automatically from initialization.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        config: dict,
        use_wandb: bool = True,
        wandb_project: str = "cs336",
        wandb_run_name: str | None = None,
    ):
        """
        Initialize the experiment logger.

        Args:
            output_dir: Directory to save logs and summaries
            experiment_name: Name for this experiment
            config: Dictionary of hyperparameters and settings
            use_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
            wandb_run_name: W&B run name (defaults to experiment_name)
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = use_wandb

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize timing
        self.start_time = time.time()
        self.start_datetime = datetime.now().isoformat()

        # Track metrics history
        self.metrics_history: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_val_step = 0
        self.last_train_loss = None
        self.last_val_loss = None

        # Initialize wandb if enabled
        if self.use_wandb:
            import wandb

            wandb.init(
                project=wandb_project,
                name=wandb_run_name or experiment_name,
                config=config,
            )

        # Save initial config
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"[ExperimentLogger] Initialized experiment: {experiment_name}")
        print(f"[ExperimentLogger] Output directory: {output_dir}")
        print(f"[ExperimentLogger] Wandb enabled: {use_wandb}")

    def get_wallclock_time(self) -> float:
        """Get seconds elapsed since training start."""
        return time.time() - self.start_time

    def get_memory_stats(self, device: str) -> dict[str, float]:
        """
        Get memory statistics in MB.

        Args:
            device: The device being used ('cuda', 'mps', or 'cpu')

        Returns:
            Dictionary with memory stats
        """
        stats = {}

        # Process memory (works for all devices)
        process = psutil.Process()
        stats["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)

        # GPU memory for CUDA
        if device == "cuda" and torch.cuda.is_available():
            stats["memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)

        return stats

    def log_step(
        self,
        step: int,
        train_loss: float | None = None,
        learning_rate: float | None = None,
        tokens_per_sec: float | None = None,
        device: str = "cpu",
        extra_metrics: dict | None = None,
    ):
        """
        Log metrics for a training step.

        Args:
            step: Current training step/iteration
            train_loss: Training loss value
            learning_rate: Current learning rate
            tokens_per_sec: Training throughput
            device: Device being used for memory stats
            extra_metrics: Additional metrics to log
        """
        wallclock = self.get_wallclock_time()
        memory_stats = self.get_memory_stats(device)

        metrics = ExperimentMetrics(
            step=step,
            wallclock_time=wallclock,
            train_loss=train_loss,
            learning_rate=learning_rate,
            tokens_per_sec=tokens_per_sec,
            **memory_stats,
        )

        if train_loss is not None:
            self.last_train_loss = train_loss

        # Convert to dict for logging
        metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}

        # Add extra metrics
        if extra_metrics:
            metrics_dict.update(extra_metrics)

        # Store in history
        self.metrics_history.append(metrics_dict)

        # Log to wandb
        if self.use_wandb:
            import wandb

            wandb_metrics = {f"train/{k}": v for k, v in metrics_dict.items()}
            wandb_metrics["train/step"] = step
            wandb_metrics["train/wallclock_time"] = wallclock
            wandb.log(wandb_metrics, step=step)

        # Print to console
        parts = [f"Step {step}"]
        if train_loss is not None:
            parts.append(f"Loss: {train_loss:.4f}")
        if learning_rate is not None:
            parts.append(f"LR: {learning_rate:.2e}")
        if tokens_per_sec is not None:
            parts.append(f"Tok/s: {tokens_per_sec:.0f}")
        parts.append(f"Time: {wallclock:.1f}s")
        if "memory_allocated_mb" in memory_stats:
            parts.append(f"GPU: {memory_stats['memory_allocated_mb']:.0f}MB")

        print(" | ".join(parts))

    def log_validation(
        self,
        step: int,
        val_loss: float,
        extra_metrics: dict | None = None,
    ):
        """
        Log validation metrics.

        Args:
            step: Current training step
            val_loss: Validation loss value
            extra_metrics: Additional validation metrics
        """
        wallclock = self.get_wallclock_time()

        self.last_val_loss = val_loss

        # Track best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_step = step

        metrics_dict = {
            "step": step,
            "wallclock_time": wallclock,
            "val_loss": val_loss,
        }

        if extra_metrics:
            metrics_dict.update(extra_metrics)

        # Store in history
        self.metrics_history.append(metrics_dict)

        # Log to wandb
        if self.use_wandb:
            import wandb

            wandb_metrics = {
                "val/loss": val_loss,
                "val/best_loss": self.best_val_loss,
                "val/wallclock_time": wallclock,
            }
            if extra_metrics:
                wandb_metrics.update({f"val/{k}": v for k, v in extra_metrics.items()})
            wandb.log(wandb_metrics, step=step)

        print(f"Step {step} | Val Loss: {val_loss:.4f} | Best: {self.best_val_loss:.4f} (step {self.best_val_step})")

    def save_metrics_history(self):
        """Save all logged metrics to a JSON file."""
        metrics_path = os.path.join(self.output_dir, "metrics_history.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"[ExperimentLogger] Saved metrics history to {metrics_path}")

    def finish(self, total_steps: int, notes: str = ""):
        """
        Finalize logging and save experiment summary.

        Args:
            total_steps: Total number of training steps completed
            notes: Optional notes about the experiment
        """
        end_time = datetime.now().isoformat()
        total_wallclock = self.get_wallclock_time()

        summary = ExperimentSummary(
            experiment_name=self.experiment_name,
            start_time=self.start_datetime,
            end_time=end_time,
            total_steps=total_steps,
            total_wallclock_seconds=total_wallclock,
            final_train_loss=self.last_train_loss,
            final_val_loss=self.last_val_loss,
            best_val_loss=self.best_val_loss if self.best_val_loss != float("inf") else None,
            best_val_step=self.best_val_step if self.best_val_loss != float("inf") else None,
            config=self.config,
            notes=notes,
        )

        # Save summary
        summary_path = os.path.join(self.output_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        # Save metrics history
        self.save_metrics_history()

        # Close wandb
        if self.use_wandb:
            import wandb

            wandb.finish()

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Total steps: {total_steps}")
        print(f"Total time: {total_wallclock / 60:.2f} minutes")
        if self.last_train_loss is not None:
            print(f"Final train loss: {self.last_train_loss:.4f}")
        if self.last_val_loss is not None:
            print(f"Final val loss: {self.last_val_loss:.4f}")
        if self.best_val_loss != float("inf"):
            print(f"Best val loss: {self.best_val_loss:.4f} (step {self.best_val_step})")
        print(f"Summary saved to: {summary_path}")
        print("=" * 60 + "\n")

        return summary
