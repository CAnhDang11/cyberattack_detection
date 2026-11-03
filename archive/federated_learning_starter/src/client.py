"""Federated Learning Client: Local Training, Optimization, and Logging."""
import os
import time
import socket
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import MLPClassifier
from src.dataset import load_federated_data
from src.network import send_msg, recv_msg, serialize_state_dict, deserialize_state_dict
from src.utils import setup_logger, compute_metrics


def train_local_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one local epoch of training on local data partition."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_local(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate local model on local validation split."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, targets in val_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total_samples": total,
    }


class FederatedClient:
    def __init__(
        self,
        server_host: str = "127.0.0.1",
        server_port: int = 8080,
        client_id: int = 0,
        num_clients: int = 2,
        data_path: Optional[str] = None,
        partition_type: str = "iid",
        dirichlet_alpha: float = 0.5,
        batch_size: int = 128,
        log_dir: str = "logs",
        device: str = "auto",
        seed: int = 42,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = client_id
        self.num_clients = num_clients
        self.data_path = data_path
        self.partition_type = partition_type
        self.dirichlet_alpha = dirichlet_alpha
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.seed = seed

        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = setup_logger(
            f"FL-Client-{client_id}",
            log_dir=self.log_dir,
            log_filename=f"client_{client_id}.log",
        )

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

        # Load local partition
        self.train_loader, self.val_loader, _, self.input_dim, self.num_classes = load_federated_data(
            data_path=self.data_path,
            client_id=self.client_id,
            num_clients=self.num_clients,
            partition_type=self.partition_type,
            alpha=self.dirichlet_alpha,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        # Local model template
        self.local_model = MLPClassifier(
            input_dim=self.input_dim,
            hidden_dims=[128, 64, 32],
            num_classes=self.num_classes,
        ).to(self.device)

        self.num_train_samples = len(self.train_loader.dataset)
        self.num_val_samples = len(self.val_loader.dataset) if self.val_loader else 0

    def start(self):
        """Connect to server and execute rounds as commanded."""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting Client {self.client_id}")
        self.logger.info(f"Connecting to Server at {self.server_host}:{self.server_port}")
        self.logger.info(f"Local Data Partition: {self.num_train_samples} train, {self.num_val_samples} val samples")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 60)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.server_host, self.server_port))
            self.logger.info(f"Successfully connected to server at {self.server_host}:{self.server_port}")

            # Handshake / Register
            send_msg(sock, {"type": "REGISTER", "client_id": f"Client_{self.client_id}"})
            self.logger.info(f"Sent registration for Client_{self.client_id}. Waiting for training to start...")

            while True:
                header, payload = recv_msg(sock)
                msg_type = header.get("type")

                if msg_type == "TRAINING_COMPLETE":
                    self.logger.info("Received TRAINING_COMPLETE from server. Shutting down gracefully.")
                    break

                elif msg_type == "ROUND_START":
                    round_num = header.get("round", 1)
                    local_epochs = header.get("local_epochs", 1)
                    lr = header.get("lr", 0.01)

                    self.logger.info(f"\n--- [Round {round_num}] Received Global Model ---")

                    # Load global weights
                    global_weights = deserialize_state_dict(payload, device=self.device)
                    self.local_model.load_state_dict(global_weights)

                    # Train locally
                    optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.9)
                    criterion = nn.CrossEntropyLoss()

                    start_time = time.time()
                    last_loss = 0.0
                    for epoch in range(1, local_epochs + 1):
                        last_loss = train_local_epoch(
                            self.local_model, self.train_loader, optimizer, criterion, self.device
                        )
                        self.logger.info(f"  Epoch {epoch}/{local_epochs} - Loss: {last_loss:.4f}")

                    train_duration = time.time() - start_time

                    # Local validation
                    val_metrics = evaluate_local(self.local_model, self.val_loader, self.device)
                    self.logger.info(
                        f"  Local Eval: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f} "
                        f"(Duration: {train_duration:.2f}s)"
                    )

                    # Serialize local weights and send back
                    local_weights_bytes = serialize_state_dict(self.local_model.state_dict())
                    result_msg = {
                        "type": "ROUND_RESULT",
                        "client_id": f"Client_{self.client_id}",
                        "num_samples": self.num_train_samples,
                        "local_loss": last_loss,
                        "duration": train_duration,
                    }
                    send_msg(sock, result_msg, local_weights_bytes)
                    self.logger.info(f"  ✓ Sent updated weights to server ({len(local_weights_bytes) / 1024:.1f} KB)")

                else:
                    self.logger.warning(f"Unknown message type received: {msg_type}")

        except ConnectionRefusedError:
            self.logger.error(f"Cannot connect to server at {self.server_host}:{self.server_port}. Is the server running?")
        except Exception as e:
            self.logger.error(f"Error during client operation: {e}", exc_info=True)
        finally:
            sock.close()
            self.logger.info("Client connection closed.")
