#!/usr/bin/env python3
"""Entry point for Federated Learning Server."""
import argparse
import sys
import os

# Ensure package root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.server import FederatedServer


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Server (FedAvg)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients required before each round starts (default: 2)")
    parser.add_argument("--rounds", type=int, default=10, help="Total number of communication rounds (default: 10)")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round on each client (default: 2)")
    parser.add_argument("--lr", type=float, default=0.01, help="Client learning rate (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and eval (default: 128)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to preprocessed data.npz (default: auto-detected)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save logs and checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for text logs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    server = FederatedServer(
        host=args.host,
        port=args.port,
        min_clients=args.min_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
    )
    server.run()


if __name__ == "__main__":
    main()
