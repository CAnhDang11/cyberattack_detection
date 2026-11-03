# HƯỚNG DẪN CHẠY FEDERATED LEARNING QUA MẠNG (2 MÁY TRAIN CÙNG LÚC)

Hệ thống cho phép **2 máy tính** (hoặc nhiều hơn) cùng tham gia huấn luyện mô hình học máy phân tán trên bộ dữ liệu **UNSW-NB15** theo thuật toán **FedAvg** (Federated Averaging).

---

## 1. Mô hình kiến trúc & Luồng hoạt động

```text
               ┌──────────────────────────────────────────────┐
               │              MÁY CHỦ (SERVER)                │
               │            Địa chỉ: 0.0.0.0:8080             │
               │  - Quản lý các vòng lặp (Communication Rounds)│
               │  - Tổng hợp trọng số (FedAvg)                │
               │  - Đánh giá mô hình toàn cục trên Test Set   │
               │  - Xuất log và lưu checkpoint                │
               └───────────────▲───────────────▲──────────────┘
                               │               │
                  (TCP Socket) │               │ (TCP Socket)
                               │               │
       ┌───────────────────────┴──┐         ┌──┴──────────────────────────┐
       │     CLIENT 0 (Máy bạn)   │         │     CLIENT 1 (Máy đối tác)  │
       │  - Nhận 50% tập train    │         │  - Nhận 50% tập train       │
       │  - Huấn luyện cục bộ SGD │         │  - Huấn luyện cục bộ SGD    │
       │  - Ghi log client_0.log  │         │  - Ghi log client_1.log     │
       └──────────────────────────┘         └─────────────────────────────┘
```

- **Mô hình**: Mạng nơ-ron `MLPClassifier` với cấu trúc `191` (features) $\rightarrow$ `128` $\rightarrow$ `64` $\rightarrow$ `32` $\rightarrow$ `2` (nhãn 0/1).
- **Dữ liệu**: Nạp trực tiếp từ `output_preprocess/data.npz`. Hệ thống sẽ tự động phân chia tập train đều cho Client 0 và Client 1.
- **Giao thức mạng**: TCP Sockets truyền nhị phân serialized PyTorch `state_dict`, có frame độ dài chống lỗi đứt đoạn gói tin.

---

## 2. Chuẩn bị trước khi chạy

### A. Xác định địa chỉ IP máy Server
- Trên máy của bạn (Server), mở terminal gõ:
  ```bash
  ifconfig | grep "inet "
  ```
  *(Ví dụ IP của bạn trong mạng LAN là `10.20.1.112`)*.

### B. Chuẩn bị trên máy đối tác (Client 1)
Máy đối tác cần:
1. Có mã nguồn thư mục `federated_learning_starter/`.
2. Có file dữ liệu `output_preprocess/data.npz` (nằm ở thư mục cha `../output_preprocess/data.npz` hoặc ngay trong `federated_learning_starter/output_preprocess/data.npz`).
3. Đã kích hoạt môi trường Python có cài `torch`, `numpy`.

---

## 3. Các bước thực hiện chi tiết

### BƯỚC 1: Khởi động Server (trên Máy bạn)
Mở **Terminal 1** trên máy bạn và chạy:

```bash
cd /Users/dangchauanh/Downloads/Cloud/federated_learning_starter
source ../.venv/bin/activate

python run_server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --min-clients 2 \
  --rounds 10 \
  --local-epochs 2 \
  --lr 0.01 \
  --batch-size 128
```
> **Trạng thái**: Server sẽ thông báo:
> `[FL-Server] Server listening on 0.0.0.0:8080`
> `[FL-Server] Waiting for at least 2 clients to register...`

---

### BƯỚC 2: Khởi động Client 0 (ngay trên Máy bạn)
Mở thêm **Terminal 2** trên máy bạn và chạy:

```bash
cd /Users/dangchauanh/Downloads/Cloud/federated_learning_starter
source ../.venv/bin/activate

python run_client.py \
  --server-host 127.0.0.1 \
  --server-port 8080 \
  --client-id 0 \
  --num-clients 2
```
> **Trạng thái**: Client 0 kết nối thành công, Server ghi nhận `(1/2 connected)`.

---

### BƯỚC 3: Khởi động Client 1 (trên Máy đối tác)
Mở terminal trên **máy đối tác** và kết nối về IP Server (`10.20.1.112`):

```bash
cd federated_learning_starter
source .venv/bin/activate

python run_client.py \
  --server-host 10.20.1.112 \
  --server-port 8080 \
  --client-id 1 \
  --num-clients 2
```
> **Trạng thái**: 
> - Khi Client 1 kết nối, Server ghi nhận đủ `(2/2 connected)`.
> - Quá trình huấn luyện Federated Learning sẽ **tự động bắt đầu ngay lập tức**!

---

## 4. Kiểm tra Log & Kết quả huấn luyện

### A. Theo dõi trực tiếp trên Terminal
Trong mỗi vòng lặp (Round), Server và các Client sẽ in rõ ràng:
- Epoch huấn luyện nội bộ của từng client.
- Thời gian huấn luyện (giây) và kích thước trọng số gửi đi.
- Kết quả FedAvg sau khi gộp trọng số.
- Đánh giá toàn cục trên tập Test: **Test Loss**, **Accuracy**, **Precision**, **Recall**, **F1-Score**.

### B. Xem các file Log & Báo cáo
Hệ thống tự động ghi nhật ký vào các thư mục sau:

1. **File Log văn bản**:
   - Server: `logs/server.log`
   - Client 0: `logs/client_0.log`
   - Client 1: `logs/client_1.log`
   *(Có thể xem realtime bằng lệnh: `tail -f logs/server.log`)*

2. **File bảng chỉ số qua các round (CSV)**:
   - `outputs/federated_history.csv`
   - Chứa đầy đủ các cột: `r◊und, clients_participated, avg_local_loss, val_loss, val_accuracy, val_precision, val_recall, val_f1, round_time_sec`.

3. **Mô hình lưu trữ (Checkpoints)**:
   - `outputs/checkpoints/best_global_model.pt`: Mô hình đạt điểm F1 / Accuracy cao nhất.
   - `outputs/checkpoints/global_model_round_<X>.pt`: Checkpoint theo từng round.

---

## 5. Xử lý sự cố thường gặp (Troubleshooting)

1. **Lỗi `ConnectionRefusedError: [Errno 61] Connection refused`**:
   - Nguyên nhân: Server chưa được bật, hoặc nhập sai cổng/IP.
   - Khắc phục: Đảm bảo Bước 1 (Server) đã chạy trước khi bật Client.

2. **Máy đối tác không kết nối được tới Server (Timeout)**:
   - Nguyên nhân: Hai máy không cùng mạng Wi-Fi/LAN, hoặc tường lửa (Firewall) chặn cổng 8080.
   - Kiểm tra: Trên máy đối tác gõ lệnh: `ping 10.20.1.112`.
   - Nếu dùng macOS làm Server: Vào `System Settings` $\rightarrow$ `Network` $\rightarrow$ `Firewall` $\rightarrow$ tạm thời tắt hoặc cho phép Python nhận kết nối.

3. **Thử nghiệm nhanh trên 1 máy (Smoke test)**:
   - Bạn có thể mở 3 terminal trên cùng máy mình:
     - Terminal 1: Chạy Server (`--min-clients 2`).
     - Terminal 2: Chạy Client 0 (`--server-host 127.0.0.1 --client-id 0`).
     - Terminal 3: Chạy Client 1 (`--server-host 127.0.0.1 --client-id 1`).
   - Cả 2 client sẽ cùng train trên máy bạn để kiểm tra toàn bộ luồng trước khi chạy với máy đối tác.
