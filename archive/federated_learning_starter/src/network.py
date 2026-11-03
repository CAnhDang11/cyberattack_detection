"""Network socket protocol and serialization helpers for Federated Learning."""
import io
import json
import struct
import socket
from typing import Tuple, Dict, Any, Optional
import torch


def send_msg(sock: socket.socket, header: Dict[str, Any], payload: Optional[bytes] = None) -> None:
    """Send a length-prefixed JSON header and optional binary payload over TCP socket."""
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)
    payload_len = len(payload) if payload else 0

    # Frame: [4 bytes header len][header json][8 bytes payload len][payload bytes]
    prefix = struct.pack(">IQ", header_len, payload_len)
    sock.sendall(prefix)
    sock.sendall(header_json)
    if payload and payload_len > 0:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Read exactly num_bytes from socket, raising ConnectionError if closed early."""
    buffer = bytearray(num_bytes)
    view = memoryview(buffer)
    pos = 0
    while pos < num_bytes:
        nbytes = sock.recv_into(view[pos:])
        if nbytes == 0:
            raise ConnectionError(f"Socket connection closed unexpectedly after receiving {pos}/{num_bytes} bytes.")
        pos += nbytes
    return bytes(buffer)


def recv_msg(sock: socket.socket) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """Receive a length-prefixed JSON header and optional binary payload from TCP socket."""
    # Read [4 bytes header len][8 bytes payload len]
    prefix = _recv_exact(sock, 12)
    header_len, payload_len = struct.unpack(">IQ", prefix)

    header_bytes = _recv_exact(sock, header_len)
    header = json.loads(header_bytes.decode("utf-8"))

    payload = None
    if payload_len > 0:
        payload = _recv_exact(sock, payload_len)

    return header, payload


def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Serialize PyTorch state_dict to bytes using io.BytesIO."""
    # Ensure tensors are detached and placed on CPU before saving for network transmission
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    buffer = io.BytesIO()
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def deserialize_state_dict(payload: bytes, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """Deserialize bytes back into PyTorch state_dict and place onto specified device."""
    buffer = io.BytesIO(payload)
    state_dict = torch.load(buffer, map_location=device)
    return state_dict
