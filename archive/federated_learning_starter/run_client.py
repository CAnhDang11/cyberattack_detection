#!/usr/bin/env python3
"""Entry point for Federated Learning Client."""
import argparse
import sys
import os

# Ensure package root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.client import FederatedClient


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--server-host", type=str, default="127.0.0.1", help="Server IP address (default: 127.0.0.1)")
    parser.add_argument("--server-port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--client-id", type=int, default=0, help="Unique ID for this client, e.g. 0 or 1 (default: 0)")
    parser.add_argument("--num-clients", type=int, default=2, help="Total number of client partitions (default: 2)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to preprocessed data.npz (default: auto-detected)")
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "dirichlet"], help="Partition strategy: iid or dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID partition (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for local training (default: 128)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for text logs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    client = FederatedClient(
        server_host=args.server_host,
        server_port=args.server_port,
        client_id=args.client_id,
        num_clients=args.num_clients,
        data_path=args.data_path,
        partition_type=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
    )
    client.start()


if __name__ == "__main__":
    main()
