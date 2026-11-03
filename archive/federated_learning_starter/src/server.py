"""Federated Learning Server: Network Coordination, FedAvg Aggregation, Evaluation & Logging."""
import os
import csv
import time
import socket
import threading
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import MLPClassifier
from src.dataset import load_federated_data
from src.network import send_msg, recv_msg, serialize_state_dict, deserialize_state_dict
from src.utils import setup_logger, compute_metrics


def fedavg(client_updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Sample-weighted average of client state_dicts (FedAvg)."""
    if not client_updates:
        raise ValueError("client_updates must not be empty")

    total_samples = sum(int(n) for _, n in client_updates)
    if total_samples <= 0:
        raise ValueError("Total number of client samples must be positive")

    ref_state = client_updates[0][0]
    aggregated = {}
    for key, ref_tensor in ref_state.items():
        if torch.is_floating_point(ref_tensor):
            acc = torch.zeros_like(ref_tensor, dtype=torch.float64)
            for state, n in client_updates:
                acc += state[key].to(torch.float64) * (int(n) / total_samples)
            aggregated[key] = acc.to(dtype=ref_tensor.dtype)
        else:
            aggregated[key] = ref_tensor.clone()
    return aggregated


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate global model on test dataset and return metrics."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for features, targets in test_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    total_samples = len(test_loader.dataset)
    avg_loss = total_loss / max(total_samples, 1)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(avg_loss)
    metrics["total_samples"] = total_samples
    return metrics


class FederatedServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        min_clients: int = 2,
        rounds: int = 10,
        local_epochs: int = 2,
        lr: float = 0.01,
        batch_size: int = 128,
        data_path: str = None,
        output_dir: str = "outputs",
        log_dir: str = "logs",
        device: str = "auto",
        seed: int = 42,
    ):
        self.host = host
        self.port = port
        self.min_clients = min_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.data_path = data_path
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.seed = seed

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = setup_logger("FL-Server", log_dir=self.log_dir, log_filename="server.log")

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # Load test dataset for global evaluation
        _, _, self.test_loader, self.input_dim, self.num_classes = load_federated_data(
            data_path=self.data_path,
            client_id=None,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        # Global model initialization
        torch.manual_seed(self.seed)
        self.global_model = MLPClassifier(
            input_dim=self.input_dim,
            hidden_dims=[128, 64, 32],
            num_classes=self.num_classes,
        ).to(self.device)

        self.clients: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.history_csv = os.path.join(self.output_dir, "federated_history.csv")
        self._init_history_csv()

    def _init_history_csv(self):
        with open(self.history_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "clients_participated", "avg_local_loss", "val_loss",
                "val_accuracy", "val_precision", "val_recall", "val_f1",
                "round_time_sec"
            ])

    def wait_for_clients(self, server_sock: socket.socket):
        """Wait until at least min_clients are connected."""
        self.logger.info(f"Server listening on {self.host}:{self.port}")
        self.logger.info(f"Waiting for at least {self.min_clients} clients to register...")

        while True:
            client_sock, client_addr = server_sock.accept()
            try:
                header, _ = recv_msg(client_sock)
                if header.get("type") == "REGISTER":
                    client_id = header.get("client_id", f"client_{len(self.clients)}")
                    with self.lock:
                        self.clients.append({
                            "id": client_id,
                            "sock": client_sock,
                            "addr": client_addr,
                        })
                        self.logger.info(
                            f"Client '{client_id}' registered from {client_addr[0]}:{client_addr[1]} "
                            f"({len(self.clients)}/{self.min_clients} connected)"
                        )
                        if len(self.clients) >= self.min_clients:
                            self.logger.info(f"All {len(self.clients)} required clients are connected! Starting FL loop.")
                            break
            except Exception as e:
                self.logger.error(f"Error during client handshake from {client_addr}: {e}")
                client_sock.close()

    def run(self):
        """Main training loop across communication rounds."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Federated Learning Server")
        self.logger.info(f"Input Dim: {self.input_dim} | Classes: {self.num_classes} | Device: {self.device}")
        self.logger.info(f"Rounds: {self.rounds} | Local Epochs: {self.local_epochs} | LR: {self.lr} | Batch: {self.batch_size}")
        self.logger.info(f"Test samples for evaluation: {len(self.test_loader.dataset)}")
        self.logger.info("=" * 60)

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(self.min_clients + 5)

        try:
            self.wait_for_clients(server_sock)

            # Evaluate initial global model before training
            init_metrics = evaluate_model(self.global_model, self.test_loader, self.device)
            self.logger.info(
                f"[Initial Eval] Loss={init_metrics['loss']:.4f} | "
                f"Acc={init_metrics['accuracy']:.4f} | "
                f"F1={init_metrics['f1']:.4f}"
            )

            best_f1 = init_metrics["f1"]
            best_acc = init_metrics["accuracy"]

            for round_idx in range(1, self.rounds + 1):
                round_start_time = time.time()
                self.logger.info(f"\n--- [Round {round_idx}/{self.rounds}] ---")

                # Serialize current global weights
                global_weights = serialize_state_dict(self.global_model.state_dict())

                # Broadcast ROUND_START to all connected clients
                client_updates = []
                update_threads = []
                thread_lock = threading.Lock()

                def communicate_with_client(client_info):
                    try:
                        cid = client_info["id"]
                        csock = client_info["sock"]

                        start_msg = {
                            "type": "ROUND_START",
                            "round": round_idx,
                            "local_epochs": self.local_epochs,
                            "lr": self.lr,
                            "batch_size": self.batch_size,
                        }
                        send_msg(csock, start_msg, global_weights)

                        # Receive client response
                        header, payload = recv_msg(csock)
                        if header.get("type") == "ROUND_RESULT":
                            num_samples = header.get("num_samples", 0)
                            local_loss = header.get("local_loss", 0.0)
                            client_time = header.get("duration", 0.0)
                            client_state = deserialize_state_dict(payload, device=torch.device("cpu"))

                            with thread_lock:
                                client_updates.append((client_state, num_samples, local_loss, cid))

                            self.logger.info(
                                f"  ✓ Received update from '{cid}': "
                                f"samples={num_samples}, local_loss={local_loss:.4f}, duration={client_time:.2f}s"
                            )
                        else:
                            self.logger.warning(f"  ✗ Unexpected message type from '{cid}': {header.get('type')}")
                    except Exception as err:
                        self.logger.error(f"  ✗ Error communicating with client '{client_info['id']}': {err}")

                for c in self.clients:
                    t = threading.Thread(target=communicate_with_client, args=(c,))
                    t.start()
                    update_threads.append(t)

                for t in update_threads:
                    t.join()

                if not client_updates:
                    self.logger.error("No valid client updates received in this round! Aborting.")
                    break

                # Aggregate weights via FedAvg
                fed_updates = [(state, n) for state, n, _, _ in client_updates]
                avg_local_loss = sum(l for _, _, l, _ in client_updates) / len(client_updates)
                new_state = fedavg(fed_updates)
                self.global_model.load_state_dict(new_state)

                # Global evaluation on test set
                eval_metrics = evaluate_model(self.global_model, self.test_loader, self.device)
                round_duration = time.time() - round_start_time

                self.logger.info(
                    f"Round {round_idx:02d} Summary: "
                    f"Avg Local Loss={avg_local_loss:.4f} | "
                    f"Test Loss={eval_metrics['loss']:.4f} | "
                    f"Acc={eval_metrics['accuracy']:.4f} | "
                    f"Precision={eval_metrics['precision']:.4f} | "
                    f"Recall={eval_metrics['recall']:.4f} | "
                    f"F1={eval_metrics['f1']:.4f} | "
                    f"Time={round_duration:.2f}s"
                )

                # Save round metric to CSV
                with open(self.history_csv, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        round_idx, len(client_updates), f"{avg_local_loss:.5f}",
                        f"{eval_metrics['loss']:.5f}", f"{eval_metrics['accuracy']:.5f}",
                        f"{eval_metrics['precision']:.5f}", f"{eval_metrics['recall']:.5f}",
                        f"{eval_metrics['f1']:.5f}", f"{round_duration:.2f}"
                    ])

                # Save latest checkpoint
                ckpt_path = os.path.join(self.output_dir, "checkpoints", f"global_model_round_{round_idx}.pt")
                torch.save({"round": round_idx, "state_dict": self.global_model.state_dict(), "metrics": eval_metrics}, ckpt_path)

                # Save best checkpoint
                if eval_metrics["f1"] >= best_f1:
                    best_f1 = eval_metrics["f1"]
                    best_acc = eval_metrics["accuracy"]
                    best_path = os.path.join(self.output_dir, "checkpoints", "best_global_model.pt")
                    torch.save({"round": round_idx, "state_dict": self.global_model.state_dict(), "metrics": eval_metrics}, best_path)
                    self.logger.info(f"  ★ New best model saved to {best_path} (F1: {best_f1:.4f}, Acc: {best_acc:.4f})")

            # Training completed, notify clients to finish
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"Training completed successfully! Best F1: {best_f1:.4f}, Best Acc: {best_acc:.4f}")
            self.logger.info("Notifying clients to terminate connections...")
            for c in self.clients:
                try:
                    send_msg(c["sock"], {"type": "TRAINING_COMPLETE"})
                    c["sock"].close()
                except Exception:
                    pass

        finally:
            server_sock.close()
            self.logger.info("Server socket closed.")
