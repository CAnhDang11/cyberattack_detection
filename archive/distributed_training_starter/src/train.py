"""Train the example classifier in single-process or DDP mode with logging and checkpointing."""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from src.dataset import build_datasets, make_loaders
from src.model import MLPClassifier
from src.distributed import (
    setup_distributed, wrap_ddp, is_main_process, cleanup_distributed, reduce_sum
)
from src.logger import setup_logger, init_metrics_csv, log_epoch_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP training starter with full logging")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per worker")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--data-path", type=str, default=None, help="Path to preprocessed data.npz")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset fallback")
    parser.add_argument("--num-samples", type=int, default=12000, help="Samples for synthetic fallback")
    parser.add_argument("--input-dim", type=int, default=191, help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for checkpoints and metrics")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, distributed):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    # totals: [weighted loss sum, correct predictions, number of examples]
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)
        totals[0] += loss.item() * labels.size(0)
        totals[1] += (logits.argmax(dim=1) == labels).sum().item()
        totals[2] += labels.size(0)

    reduce_sum(totals, distributed)
    count = max(totals[2].item(), 1.0)
    return totals[0].item() / count, totals[1].item() / count


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")

    info = setup_distributed()
    distributed = info["distributed"]
    rank = info["rank"]
    local_rank = info["local_rank"]
    world_size = info["world_size"]
    device = info["device"]

    if args.device == "cpu":
        if distributed and device.type == "cuda":
            raise RuntimeError("For CPU DDP, launch on a machine without CUDA or unset CUDA visibility.")
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")

    # Setup logger
    logger = setup_logger(
        name="DDP",
        log_dir=args.log_dir,
        log_filename="distributed_train.log",
        rank=rank,
    )

    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    train_ds, val_ds, actual_input_dim, actual_num_classes = build_datasets(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        seed=args.seed,
        data_path=args.data_path,
        force_synthetic=args.synthetic,
    )
    train_loader, val_loader, train_sampler = make_loaders(
        train_ds, val_ds, batch_size=args.batch_size,
        rank=rank, world_size=world_size, distributed=distributed,
        num_workers=args.num_workers,
    )

    model = MLPClassifier(
        input_dim=actual_input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=actual_num_classes,
    ).to(device)
    model = wrap_ddp(model, device, distributed, local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    history_csv = None
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        history_csv = init_metrics_csv(args.output_dir)
        logger.info("=" * 65)
        logger.info("Starting Distributed Data Parallel (DDP) Training")
        logger.info(f"World Size: {world_size} workers | Device: {device} | Distributed: {distributed}")
        logger.info(f"Features: {actual_input_dim} | Classes: {actual_num_classes}")
        logger.info(f"Train samples: {len(train_ds):,} | Validation samples: {len(val_ds):,}")
        logger.info(f"Epochs: {args.epochs} | Batch Size per worker: {args.batch_size} | LR: {args.lr}")
        logger.info(f"Log file: {os.path.join(args.log_dir, 'distributed_train.log')}")
        logger.info("=" * 65)

    best_val_acc = 0.0
    start_train_time = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_totals = torch.zeros(3, dtype=torch.float64, device=device)
            for features, labels in train_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_totals[0] += loss.item() * labels.size(0)
                train_totals[1] += (logits.argmax(dim=1) == labels).sum().item()
                train_totals[2] += labels.size(0)

            reduce_sum(train_totals, distributed)
            train_count = max(train_totals[2].item(), 1.0)
            train_loss = train_totals[0].item() / train_count
            train_acc = train_totals[1].item() / train_count

            val_loss, val_acc = evaluate(model, val_loader, criterion, device, distributed)
            epoch_duration = time.time() - epoch_start

            if is_main_process(rank):
                logger.info(
                    f"Epoch {epoch:03d}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                    f"Time: {epoch_duration:.2f}s"
                )

                if history_csv:
                    log_epoch_metrics(history_csv, epoch, epoch_duration, train_loss, train_acc, val_loss, val_acc)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    raw_model = model.module if hasattr(model, "module") else model
                    ckpt_path = output_dir / "best.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": val_acc,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                            "args": vars(args),
                        },
                        ckpt_path,
                    )
                    logger.info(f"  ★ New best model checkpoint saved to: {ckpt_path} (Val Acc: {val_acc*100:.2f}%)")

        total_duration = time.time() - start_train_time
        if is_main_process(rank):
            logger.info("=" * 65)
            logger.info(f"Training completed in {total_duration:.2f}s! Best Validation Accuracy: {best_val_acc*100:.2f}%")
            logger.info(f"Best model checkpoint: {output_dir / 'best.pt'}")
            logger.info(f"Metrics CSV history: {history_csv}")
            logger.info(f"Text logs: {os.path.join(args.log_dir, 'distributed_train.log')}")
            logger.info("=" * 65)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
