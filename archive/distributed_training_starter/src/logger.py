"""Logging utilities for Distributed Training (DDP)."""
import os
import csv
import logging
from typing import Optional


class RankFilter(logging.Filter):
    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


def setup_logger(
    name: str = "DDP",
    log_dir: str = "logs",
    log_filename: str = "distributed_train.log",
    rank: int = 0,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup a logger with rank prefix, logging to both console and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(f"{name}_rank_{rank}")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [Rank %(rank)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        rank_filter = RankFilter(rank=rank)

        # File Handler (all ranks or main rank can write)
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(rank_filter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(rank_filter)
        logger.addHandler(console_handler)

    return logger


def init_metrics_csv(output_dir: str, filename: str = "training_history.csv") -> str:
    """Initialize CSV file for saving epoch metrics."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "duration_sec", "train_loss", "train_acc", "val_loss", "val_acc"])
    return csv_path


def log_epoch_metrics(csv_path: str, epoch: int, duration: float, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
    """Append one epoch of metrics to the CSV file."""
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{duration:.2f}",
            f"{train_loss:.5f}",
            f"{train_acc:.5f}",
            f"{val_loss:.5f}",
            f"{val_acc:.5f}",
        ])
