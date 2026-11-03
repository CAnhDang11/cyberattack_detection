# HƯỚNG DẪN HUẤN LUYỆN PHÂN TÁN (PYTORCH DDP) — TRÊN 1 MÁY & ĐA MÁY

Hệ thống huấn luyện phân tán sử dụng **PyTorch DDP (`DistributedDataParallel`)** và trình khởi chạy chuẩn **`torchrun`**. 

Mỗi worker là một tiến trình (process) độc lập chạy song song, nạp một phần dữ liệu không trùng lặp thông qua `DistributedSampler`, và **đồng bộ gradient (All-Reduce)** sau mỗi mini-batch.

---

## 1. Kiến trúc luồng huấn luyện

```text
       ┌───────────────────────────────┐           ┌───────────────────────────────┐
       │     MÁY 1 (Master Node)       │           │     MÁY 2 (Worker Node)       │
       │   IP: 10.20.1.112             │  All-     │   IP: Máy đối tác             │
       │   node_rank = 0               │  Reduce   │   node_rank = 1               │
       │   Dữ liệu: 50% tập train      │◄─────────►│   Dữ liệu: 50% tập train      │
       │   Cổng rendezvous: 29500      │           │   Kết nối tới: 10.20.1.112    │
       └───────────────────────────────┘           └───────────────────────────────┘
```

---

## 2. PHẦN 1: Chạy nhiều Worker / Process trên CÙNG 1 MÁY (Single-Node)

Khi chạy trên một máy tính duy nhất, cờ `--standalone` sẽ tự động cấu hình môi trường mạng cục bộ.

### Trường hợp 1: Máy có nhiều GPU (Multi-GPU)
Nếu máy bạn có 2 GPU NVIDIA:
```bash
cd distributed_training_starter
source ../.venv/bin/activate

# Lệnh 1 dòng:
torchrun --standalone --nproc_per_node=2 -m src.train --epochs 5 --batch-size 128
```

### Trường hợp 2: Máy dùng CPU / Apple Silicon Mac (Multi-Process CPU)
Nếu máy không có CUDA GPU và bạn muốn tận dụng đa nhân CPU để chạy 2 hoặc nhiều worker song song:
```bash
cd distributed_training_starter
source ../.venv/bin/activate

# Lệnh 1 dòng (đã cấu hình IPv4 cho macOS):
GLOO_SOCKET_IFNAME=lo0 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 --nproc_per_node=2 -m src.train --device cpu --epochs 2
```

> [!NOTE]
> - `--nproc_per_node`: Số tiến trình huấn luyện DDP chính (mỗi process tính gradient riêng).
> - `--num-workers` (trong `src.train`): Số luồng nạp dữ liệu từ đĩa của DataLoader.

---

## 3. PHẦN 2: Chạy giữa HAI MÁY (Multi-Node DDP)

Cấu hình cho kịch bản gồm **2 máy** kết nối qua mạng LAN / Wi-Fi:
- **`--nnodes=2`**: Tổng số máy vật lý tham gia.
- **`--nproc_per_node=1`**: Số worker chạy trên mỗi máy.
- **`--master_addr=10.20.1.112`**: Địa chỉ IP của Máy 1 (Master Node).
- **`--master_port=29500`**: Cổng đồng bộ trạng thái ban đầu (rendezvous).
- **`--node_rank`**: Định danh từng máy (`0` cho Master, `1` cho Worker tiếp theo).

---

### Bước 1: Khởi động trên Máy 1 (Master Node — IP `10.20.1.112`)

Mở Terminal trên Máy 1:
```bash
cd /Users/dangchauanh/Downloads/Cloud/distributed_training_starter
source ../.venv/bin/activate

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr=10.20.1.112 --master_port=29500 -m src.train --epochs 5 --device cpu
```
> **Trạng thái**: Máy 1 sẽ khởi tạo nhóm tiến trình phân tán và ở trạng thái chờ Máy 2 bắt tay.

---

### Bước 2: Khởi động trên Máy 2 (Worker Node)

Mở Terminal trên Máy 2 (yêu cầu cùng code và môi trường python):
```bash
cd distributed_training_starter
source .venv/bin/activate

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --master_addr=10.20.1.112 --master_port=29500 -m src.train --epochs 5 --device cpu
```
> **Trạng thái**: 
> - Khi Máy 2 kết nối, cả hai máy sẽ lập tức đồng bộ trọng số khởi tạo ban đầu.
> - Quá trình huấn luyện diễn ra đồng bộ qua từng epoch, All-Reduce gradient và in log kết quả.
> - Máy 1 (rank 0) sẽ tự động lưu mô hình tốt nhất vào `outputs/best.pt`.

---

## 4. Các tham số tùy chỉnh trong `src.train`

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--epochs` | `10` | Số lượng epoch huấn luyện |
| `--batch-size` | `128` | Kích thước batch trên mỗi worker |
| `--lr` | `0.001` | Tốc độ học (Learning Rate cho Adam) |
| `--device` | `auto` | Thiết bị tính toán: `cpu`, `cuda`, hoặc `auto` |
| `--num-samples` | `12000` | Số lượng mẫu dữ liệu |
| `--output-dir` | `outputs` | Thư mục lưu checkpoint `best.pt` |

---

## 5. Những lưu ý quan trọng & Khắc phục sự cố

1. **Giao thức Backend (`gloo` vs `nccl`)**:
   - Khi huấn luyện đa máy trên CPU hoặc giữa các máy khác nhau (như Mac, Windows, Linux không GPU đồng bộ), mã nguồn trong [src/distributed.py](src/distributed.py) đã tự động chọn backend **`gloo`**.
   - Backend `nccl` chỉ áp dụng khi toàn bộ các máy đều có GPU NVIDIA chạy Linux có cài đặt CUDA.

2. **Kiểm tra thông mạng (Ping & Port 29500)**:
   - Trên Máy 2, hãy kiểm tra kết nối tới Máy 1 bằng lệnh:
     ```bash
     ping 10.20.1.112
     ```
   - Nếu bị timeout hoặc từ chối kết nối, kiểm tra Tường lửa (Firewall) trên Máy 1:
     - Trên macOS: Vào `Cài đặt hệ thống` $\rightarrow$ `Mạng` $\rightarrow$ `Tường lửa` và tạm tắt hoặc cho phép Python kết nối đến.

3. **Phân chia dữ liệu tự động (`DistributedSampler`)**:
   - Mỗi máy sẽ chỉ nạp đúng một nửa dữ liệu ($1 / \text{world\_size}$) nhờ `DistributedSampler`, đảm bảo không trùng lặp dữ liệu giữa 2 máy.

---

## 6. Theo dõi Log & Kết quả huấn luyện

Hệ thống ghi nhận toàn diện và đồng thời ra 3 kênh:

1. **Màn hình Terminal**:
   - Gắn tag định danh worker theo rank: `[Rank 0]`, `[Rank 1]`.
   - In chi tiết từng epoch: Train Loss, Train Acc, Val Loss, Val Acc, Thời gian thực hiện (giây).

2. **File nhật ký văn bản (`logs/distributed_train.log`)**:
   - Lưu trữ toàn bộ lịch sử chạy và thông báo cấu hình mạng của các workers.
   - Theo dõi realtime:
     ```bash
     tail -f logs/distributed_train.log
     ```

3. **Bảng tổng hợp chỉ số (`outputs/training_history.csv`)**:
   - Ghi lại các cột theo từng epoch: `epoch, duration_sec, train_loss, train_acc, val_loss, val_acc`.
   - Dễ dàng mở bằng Excel hoặc vẽ đồ thị loss/acc.

4. **Mô hình tốt nhất (`outputs/best.pt`)**:
   - Tự động lưu khi Validation Accuracy đạt mức cao nhất qua các epoch.
