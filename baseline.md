# Baseline centralized — UNSW-NB15, phân loại `attack_cat` 10 class

Tài liệu này mô tả baseline centralized đã chốt: dữ liệu được chuẩn bị thế nào, model nào, kết quả ra sao, cách chạy lại, và code đã thay đổi gì so với repo gốc. Phần federated (FedAvg/FedProx) **phải dùng đúng dữ liệu và cấu hình ở đây** để so sánh được (xem [mục 7](#7-quy-tắc-cho-phần-federated)).

## 1. Tóm tắt

| | |
|---|---|
| Bài toán | Phân loại đa lớp `attack_cat`: Normal + 9 loại tấn công (10 class gốc, không gộp) |
| Dữ liệu | 4 file raw `UNSW-NB15_1..4.csv` (2.540.047 dòng), **chia lại** 70/10/20, không dùng bộ chia sẵn |
| Feature | 38 feature gốc (bỏ IP/port/thời gian và 3 feature TTL) + 9 feature tỉ lệ → 208 chiều sau one-hot |
| Model chính | **XGBoost "E8"**: 500 cây, depth 10, lr 0.1, `hist`, class weight α=0.5 |
| Baseline cho fed | **torch_mlp**: 512-256-128-64, Adam lr 1e-3, batch 256, 30 epoch, class weight α=0.25 |
| Đánh giá | P/R/F1 **từng class** trên test, trung bình 3 cách chia (seed 42/43/44); chọn mọi thứ trên validation |

XGBoost E8 là model tốt nhất trong các model centralized đã thử (logistic, sklearn MLP, torch MLP, random forest, XGBoost). torch_mlp kém hơn XGBoost nhưng có **cùng kiến trúc, optimizer, loss với model của fed**, nên là mốc đúng để đo cái giá của học phân tán.

## 2. Dữ liệu

### 2.1. Nguồn và làm sạch (`unsw/raw.py`)
- Raw: `dataset/UNSW-NB15_1.csv … _4.csv` (không header). Tên cột lấy từ `NUSW-NB15_features.csv` ở gốc repo. Có thể tải raw từ Kaggle `mrwellsdavid/unsw-nb15`.
- Làm sạch, **không xoá dòng, không đổi nhãn**:
  - `attack_cat`: bỏ khoảng trắng, ô trống → `Normal`, `Backdoors` → `Backdoor`.
  - `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd`: ô trống nghĩa là "không có HTTP/FTP" → 0 (giống bộ chia sẵn).
- Đổi tên cột cho khớp bộ chia sẵn (`smeansz`→`smean`, `Sintpkt`→`sinpkt`, `res_bdy_len`→`response_body_len`, …).

### 2.2. Feature
| Nhóm | Cột | Lý do |
|---|---|---|
| Bỏ (metadata) | `srcip, sport, dstip, dsport, stime, ltime` | Định danh testbed, không phải hành vi: 100% tấn công đến từ `175.45.176.0/24` |
| Bỏ (TTL) | `sttl, dttl, ct_state_ttl` | Gần như là proxy của máy gửi (TTL 254/62 vs 31) — shortcut đã biết của UNSW-NB15. Bỏ đi gần như không mất điểm (thử nghiệm E6) và tổng quát hơn khi mỗi client fed là một mạng khác |
| Thêm (tỉ lệ) | `byte_ratio, pkt_ratio, mean_ratio, payload_est, load_ratio, jit_ratio, intpkt_ratio, srv_src_share, srv_dst_share` | Tính trên từng dòng (không dùng nhãn, không dùng dòng khác). Shellcode tăng ổn định qua 3 cách chia (val +0.046, test +0.016 F1), không class nào giảm (E7, E8) |

Còn 47 cột (38 số/categorical gốc + 9 tỉ lệ). Tiền xử lý (`Preprocessor` trong `unsw/data.py`, **fit chỉ trên train**): điền median, `log1p`, z-score; one-hot `proto/state/service` giữ **mọi** category của train (category lạ ở val/test → toàn 0). Kết quả: 208 chiều.

### 2.3. Cách chia
- 70% train / 10% validation / 20% test.
- **Chia theo nhóm feature vector giống hệt**: các dòng có cùng vector feature luôn vào cùng một tập → không có vector nào nằm ở 2 tập. Lý do: 513k dòng trùng hoàn toàn; chia ngẫu nhiên theo dòng thì test chứa bản sao của train (rò rỉ, điểm số ảo).
- Stratify theo nhãn gốc đa số của mỗi nhóm → class nhỏ có mặt ở cả 3 tập.
- Phép chia luôn tính trên 41 feature gốc (trước khi bỏ TTL / thêm tỉ lệ) → mọi biến thể feature và mọi scheme nhãn có **cùng** train/val/test.
- Tất định: cùng raw + cùng code + cùng seed → cùng split, trên mọi máy (đã kiểm chứng giữa máy local và Kaggle).

Kích thước và dấu kiểm tra (`sha256` của `'\n'.join(test_ids)`, 16 ký tự đầu):

| seed | train | val | test | `input_dim` | sha `test_ids` |
|---|---|---|---|---|---|
| 42 | 1.779.331 | 253.637 | 507.079 | 208 | `9d6f04c123a6ca10` |
| 43 | 1.778.175 | 252.723 | 509.149 | 208 | `0e7da2dbd50ed912` |
| 44 | 1.780.056 | 254.197 | 505.794 | 208 | `82955f63607dc8cd` |

Số mẫu từng class (seed 42):

| class | train | test |
|---|---|---|
| Normal | 1.555.004 | 442.155 |
| Analysis | 1.894 | 524 |
| Backdoor | 1.630 | 461 |
| DoS | 11.388 | 3.271 |
| Exploits | 31.099 | 8.920 |
| Fuzzers | 17.016 | 4.817 |
| Generic | 150.564 | 43.387 |
| Reconnaissance | 9.546 | 3.213 |
| Shellcode | 1.065 | 298 |
| Worms | 125 | 33 |

### 2.4. Chất lượng dữ liệu (ảnh hưởng đến cách đọc kết quả)
Chi tiết: `python -m unsw.raw analyze` → `experiments/raw_analysis/REPORT.md`.
- 2.590 feature vector mang **nhiều nhãn khác nhau** (42.322 dòng); 85% là flow một chiều không phản hồi. Ở mức flow, các dòng này không thể phân biệt → Analysis, Backdoor, DoS bị trộn với Exploits. Recall của "oracle đa số" (luôn đoán nhãn phổ biến nhất của vector) chỉ là Analysis 0.23, Backdoor 0.16, DoS 0.58, Exploits 0.88. Mỗi run ghi con số này vào cột `max_recall` của `per_class.csv`. Đây **không phải** giới hạn trên của từng class: có class weight, model có thể vượt (DoS đạt 0.71).
- ~36k dòng Normal phát ra từ máy tấn công (`175.45.176.x`) giống hệt Fuzzers → nguồn chính của nhầm lẫn Fuzzers↔Normal. Normal từ máy thường: XGBoost đúng gần 100%.
- Bản ghi trùng **không xoá**: phần lớn là sự kiện thật (DNS lặp lại, `stime` chỉ chính xác tới giây); xoá sẽ làm méo Generic. Chia theo nhóm đã chặn rò rỉ.

## 3. Model

### 3.1. Class weight (dùng chung cho mọi model và cho fed)
`w_c = (N / (K·n_c))^α`, chuẩn hoá để tổng trọng số = N, tính **một lần** từ nhãn train toàn cục (`unsw.data.class_weights`). Áp dụng qua `sample_weight` (XGBoost) hoặc `cross_entropy(weight=...)` (torch, fed). α=0 là không trọng số, α=1 là cân bằng hoàn toàn.

### 3.2. XGBoost E8 (model chính)
`objective=multi:softprob`, `n_estimators=500`, `max_depth=10`, `learning_rate=0.1`, `tree_method=hist`, α=0.5, các tham số khác mặc định. XGBoost tất định → chỉ thay đổi cách chia (seed), không cần lặp seed model.

α=0.5 được chọn trên validation (thử 0 / 0.25 / 0.5 / 0.75 / 1): không class nào giảm quá 0.01 F1 val, Fuzzers/Shellcode/Worms tốt lên. Đã thử random search 20 cấu hình (B1): cấu hình tốt nhất (`b1_02`: 800 cây, α=0.75, …) chênh E8 ≤ 0.02 F1 mọi class trên test và chủ yếu chỉ dịch điểm cân bằng báo nhầm ↔ bỏ sót → giữ E8 vì đơn giản hơn.

### 3.3. torch_mlp (baseline cho fed)
`unsw.model.model`: MLP 4 lớp ẩn 512-256-128-64, ReLU, 10 output. Adam lr 1e-3, không weight decay, batch 256, 30 epoch; sau mỗi epoch đánh giá val macro-F1 và giữ checkpoint tốt nhất. α=0.25 chọn trên val (thử 0 / 0.25 / 0.5; MLP chịu trọng số mạnh kém hơn cây: precision của class nhỏ tụt nhanh khi α tăng).

## 4. Kết quả (test, trung bình 3 cách chia, Kaggle GPU T4)

| class | n/seed | XGB E8 P | R | F1 | torch_mlp P | R | F1 |
|---|---|---|---|---|---|---|---|
| Normal | 442.891 | 0.998 | 0.993 | **0.996** ±0.000 | 0.999 | 0.990 | 0.994 ±0.000 |
| Analysis | 539 | 0.233 | 0.102 | 0.142 ±0.005 | 0.237 | 0.163 | **0.168** ±0.062 |
| Backdoor | 467 | 0.219 | 0.221 | **0.220** ±0.013 | 0.429 | 0.130 | 0.166 ±0.039 |
| DoS | 3.261 | 0.357 | 0.714 | **0.476** ±0.005 | 0.344 | 0.667 | 0.453 ±0.005 |
| Exploits | 8.908 | 0.775 | 0.639 | **0.701** ±0.007 | 0.714 | 0.631 | 0.669 ±0.009 |
| Fuzzers | 4.851 | 0.557 | 0.754 | **0.641** ±0.005 | 0.493 | 0.761 | 0.599 ±0.003 |
| Generic | 43.189 | 0.998 | 0.986 | **0.992** ±0.000 | 0.997 | 0.982 | 0.989 ±0.001 |
| Reconnaissance | 2.899 | 0.923 | 0.777 | **0.844** ±0.020 | 0.887 | 0.786 | 0.834 ±0.016 |
| Shellcode | 300 | 0.623 | 0.727 | **0.670** ±0.020 | 0.387 | 0.747 | 0.507 ±0.044 |
| Worms | 35 | 0.500 | 0.458 | **0.477** ±0.020 | 0.249 | 0.410 | 0.296 ±0.086 |
| Normal bị báo là tấn công | | | | 3.033 | | | 4.269 |
| Tấn công bị đoán là Normal | | | | 838 | | | 585 |

(± là độ lệch chuẩn F1 qua 3 cách chia; 2 dòng cuối là số dòng trung bình mỗi tập test.)

Góc nhìn 7 class (quy dự đoán 10 class về: Analysis/Backdoor/DoS/Exploits → `Exploit-like`): Exploit-like F1 0.904 (XGB) / 0.883 (MLP), các class khác giữ nguyên. Train thẳng 7 class cho cùng kết quả, nên train 10 class không mất gì.

Đọc kết quả:
- Normal, Generic gần như hoàn hảo. Analysis, Backdoor, DoS thấp chủ yếu do dữ liệu (mục 2.4), không phải do model.
- Fuzzers bị giới hạn bởi Normal từ máy tấn công: thử đổi feature, class weight, ngưỡng đều chỉ trượt trên đường đánh đổi precision↔recall.
- Worms chỉ ~35 mẫu test: ±2 mẫu đã đổi F1 vài phần trăm → luôn báo kèm số mẫu.
- XGBoost hơn MLP rõ nhất ở Shellcode (−0.16), Worms (−0.18), Fuzzers (−0.04).
- **Không so trực tiếp với paper dùng bộ chia sẵn**: bộ chia sẵn có train/test lệch phân phối và vector trùng giữa 2 tập (với bộ chia sẵn, 24,5% Normal bị báo là tấn công; chia lại: 0,7%).

Kết quả đầy đủ (P/R/F1/TP từng class, từng seed, val và test) + confusion matrix: `experiments/final_report/10class/` (`per_class_all.csv`, `f1_test_10class.csv`, `confusion_*.png`). Thư mục `experiments/` không nằm trong git.

## 5. Cách chạy

Chạy từ gốc repo. Môi trường: `.venv` với numpy, pandas, scikit-learn, joblib, torch, xgboost, matplotlib, pytest.

> **macOS**: không nạp torch và xgboost trong cùng một process (crash không báo lỗi). Các lệnh dưới đây đã là process riêng.

### 5.1. Chuẩn bị dữ liệu (một lần mỗi seed)
```bash
source .venv/bin/activate
for s in 42 43 44; do
  python -m unsw.run prepare --raw-dir dataset --label-scheme original \
    --drop-features sttl,dttl,ct_state_ttl --derived-features all \
    --seed $s --output experiments/data_s$s
done
```
~1–2 phút/seed, RAM đỉnh ~9 GB. Kiểm tra `experiments/data_s<seed>/audit.json` (`fit_rows`, `validation_rows`, `test_rows`, `input_dim`) với bảng ở mục 2.3. Kiểm tra sha:
```bash
python -c "import numpy as np,hashlib,sys; z=np.load(sys.argv[1]); print(hashlib.sha256('\n'.join(z['test_ids']).encode()).hexdigest()[:16])" experiments/data_s42/data.npz
```
`data.npz` ~142 MB/seed → **không đưa lên git** (GitHub chặn file > 100 MB); tạo lại từ raw là ra y hệt.

Chạy thử nhanh: thêm `--limit-per-class 2000` (chỉ giới hạn train mỗi class; val/test giữ nguyên; không dùng để báo cáo).

### 5.2. XGBoost E8
```bash
python -m unsw.run baseline --baseline-model xgboost \
  --data experiments/data_s42/data.npz --output experiments/xgb_e8_s42 \
  --n-estimators 500 --max-depth 10 --tree-lr 0.1 --class-weight-alpha 0.5 \
  --seed 42 --n-jobs -1          # GPU: thêm --xgb-device cuda
```
`--n-jobs -1` dùng mọi nhân CPU (mặc định 1 luồng, chậm). Kết quả GPU có thể lệch rất nhẹ so với CPU (Worms ±2 mẫu) → chỉ so sánh các run cùng thiết bị.

### 5.3. torch_mlp
```bash
python -m unsw.run centralized \
  --data experiments/data_s42/data.npz --output experiments/mlp_s42 \
  --class-weight-alpha 0.25 --epochs 30 --batch-size 256 \
  --hidden 512 --hidden-layers 4 --lr 0.001 --seed 42   # GPU: thêm --device cuda
```
~3,5 phút trên CPU local (Mac mini), ~8 phút trên Kaggle T4 (model nhỏ, GPU không nhanh hơn). Thời gian train chỉ so sánh trong cùng thiết bị.

### 5.4. Kaggle (GPU, cách đã dùng cho số liệu ở mục 4)
- Notebook `kaggle/kaggle_run.ipynb`: tự tìm raw trong `/kaggle/input`, prepare đúng cấu hình trên, train, chấm từng class, vẽ confusion matrix, gói `results_<exp>_s<seed>.zip`.
- **Chạy local** (không cần CLI): tạo notebook trên Kaggle từ `kaggle/kaggle_run.ipynb`, *Add Input* dataset `mrwellsdavid/unsw-nb15`, upload code (`zip -r kaggle_code.zip unsw NUSW-NB15_features.csv -x '*/__pycache__/*'`) thành dataset của bạn rồi *Add Input*, chọn GPU T4, đặt `EXPERIMENT` ở cell cấu hình.
- **Chạy bằng script** (`.venv/bin/pip install kaggle`; đăng nhập: `cp .env.example .env` rồi điền `KAGGLE_API_TOKEN` lấy ở Kaggle → Settings → API → Generate New Token — `.env` đã gitignore, script tự nạp; hoặc để token trong `~/.kaggle/`): mỗi người đẩy lên **tài khoản Kaggle của chính mình**. `kaggle/*-metadata.json` dùng chỗ giữ `{user}`; `run_exp.sh` thay bằng `KAGGLE_USER` (nếu đặt) hoặc username của Kaggle CLI, tự tạo dataset code private `<user>/unsw-ids-code` ở lần chạy đầu.
- `zsh kaggle/run_exp.sh --exp e3_10class --seed 42 --config kaggle/e8_config.json` (XGB E8; đổi `--seed 43/44` cho 2 cách chia còn lại), `zsh kaggle/run_exp.sh --exp mlp_10class` (torch_mlp: α sweep seed 42, rồi seed 43/44), `zsh kaggle/wait_all.sh` (chờ và tải kết quả).
- Kaggle cho tối đa 2 phiên GPU cùng lúc (`run_exp.sh --retry` chờ slot trống).
- Các mode `b1`, `b1_10class`, `c2b_8class` của notebook dùng cờ đã bị xoá (mục 8.3) → không chạy lại được.

### 5.5. Test
```bash
.venv/bin/python -m pytest tests -q     # 21 pass, 2 skip (cần Flower)
```

### 5.6. Phân tích dữ liệu raw (tuỳ chọn)
```bash
python -m unsw.raw analyze --raw-dir dataset --output experiments/raw_analysis
```
Ra `REPORT.md`, `class_stats.csv` (trùng lặp, xung đột nhãn, `max_recall_*` theo class), `conflict_matrix.csv`, `class_by_file.csv`, `source_subnet.csv`, `summary.json`.

## 6. File đầu ra

Thư mục `prepare`:
| file | nội dung |
|---|---|
| `data.npz` | `X_train/val/test` (float32, 208 cột), `y_train/val/test`, `y_orig_*` + `orig_names` (nhãn 10 class gốc), `train_ids/val_ids/test_ids` (`UNSW-NB15_<n>:<dòng>`) |
| `labels.json` | tên class → chỉ số; model đọc file này để biết số output |
| `preprocess.joblib` | `Preprocessor` đã fit trên train |
| `audit.json` | cấu hình, feature dùng/bỏ, số dòng, số mẫu từng class mỗi tập |

Thư mục mỗi run:
| file | nội dung |
|---|---|
| `per_class.csv` | precision, recall, F1, support, `max_recall` (oracle đa số) từng class trên **test** |
| `confusion_matrix.csv` | hàng = thật, cột = dự đoán (test) |
| `results.json` / `results.csv` | accuracy, macro/weighted, ROC-AUC, `max_accuracy`, class weight, thời gian train, val macro-F1 đã chọn |
| `history.csv` | train loss, val loss, val macro-F1 theo epoch/round |
| `original_breakdown.csv` | nhãn gốc → class dự đoán (hữu ích khi train scheme gộp) |
| `training.log`, `config.json`, `labels.json`, `clients.csv`, `partitions.npz` | log, cấu hình, phân bổ client |
| `baseline.joblib` + `estimator_params.json` (XGB) / `best.pt` (MLP) | model |

## 7. Quy tắc cho phần federated
Để kết quả fed so được với baseline:
1. Dùng **đúng** `data.npz` của mục 5.1 (cùng seed 42/43/44, kiểm tra sha `test_ids`). Chia client bằng `unsw.data.partition` trên `y_train`; không đụng val/test.
2. Cùng model (`unsw.model.model`, 512-256-128-64), Adam lr 1e-3, batch 256, cross-entropy với **class weight α=0.25 tính từ nhãn train toàn cục** (`--class-weight-alpha 0.25` — `unsw/federated.py` đã truyền cùng trọng số này cho mọi client).
3. Chọn round/hyperparameter trên val, báo cáo test theo **từng class** (`per_class.csv`), so với cột torch_mlp ở mục 4; XGBoost E8 là mức trần centralized.
4. So sánh thời gian chỉ trong cùng thiết bị (baseline torch_mlp chạy trên Kaggle T4).
5. Hiện trạng: `unsw/run.py` báo lỗi khi `fedavg/fedprox` với `--device cuda` (runner Flower local chỉ CPU); Flower chưa cài trong `.venv`.

## 8. Thay đổi so với code gốc

Code gốc: phân loại 10 class trên bộ chia sẵn `UNSW_NB15_training-set/testing-set.csv` (không có trên máy), số class cố định 10, không class weight.

### 8.1. File mới
- `unsw/raw.py`: đọc + làm sạch raw, `analyze` (trùng lặp, xung đột nhãn, recall oracle đa số, subnet nguồn), chia theo nhóm (`group_split`), `prepare_raw`, feature tỉ lệ (`add_derived`), bỏ feature (`--drop-features`).
- `NUSW-NB15_features.csv`: tên cột raw.
- `tests/test_raw.py`: làm sạch, chia không chung vector, `analyze`, scheme gộp giữ nguyên split, biến thể feature giữ nguyên split.
- `tests/test_loader.py`: `TensorLoader` cho ra đúng các batch như `DataLoader`.

### 8.2. File sửa
| file | thay đổi |
|---|---|
| `unsw/run.py` | `prepare --raw-dir` (+ `--test-size`, `--val-size`, `--label-scheme`, `--drop-features`, `--derived-features`); `--class-weight-alpha`; `--xgb-device`; `--n-jobs`; số class lấy từ `labels.json`; ghi `max_recall`/`max_accuracy`, `original_breakdown.csv`, class weight vào kết quả; centralized dùng `TensorLoader` |
| `unsw/data.py` | scheme nhãn (`original`, `merged`) — chỉ là ánh xạ, nhãn gốc luôn lưu ở `y_orig_*`; `class_weights`; `encode/decode/load/partition/save_partitions` theo số class của dữ liệu; **sửa lỗi one-hot**: trước đây bỏ cột category đầu tiên, nên category lạ trùng với category đầu → giờ giữ mọi category |
| `unsw/baseline.py` | số class theo dữ liệu; class weight qua `sample_weight`; XGBoost `tree_method=hist` + `device`; RF/XGB dùng `--n-jobs` |
| `unsw/model.py` | `model(..., n_classes)`; `TensorLoader` (cắt batch từ tensor, cùng thứ tự ngẫu nhiên với `DataLoader` → kết quả trùng từng bit, nhanh hơn ~30% CPU, ~2× GPU); `train_epoch(..., weight=)` |
| `unsw/federated.py` | model fed theo số class của dữ liệu; cùng class weight toàn cục; `TensorLoader`. Thuật toán FedAvg/FedProx không đổi |
| `unsw/evaluation.py` | metric/bảng theo K class; `recall_ceiling` + `add_ceiling` (`max_recall`, `max_accuracy`); `original_breakdown` |
| `unsw/compare.py` | thêm cột `max_accuracy` |
| `tests/test_multiclass.py` | test one-hot đã sửa, scheme `merged`, class weight, smoke 7 class |
| `.gitignore` | bỏ qua `experiments/`, `dataset/*.csv`, `*.zip`, `/kaggle/`, log |

### 8.3. Đã thử rồi xoá khỏi code (2026-11-18)
- `--dedup` (xoá bản ghi trùng): xoá sẽ mất sự kiện thật, méo Generic; chia theo nhóm đã đủ chặn rò rỉ.
- Scheme `merged3` (8 class, gộp Analysis/Backdoor/Exploits, DoS riêng): chỉ là thử nghiệm phụ.
- `--xgb-params` (tham số XGBoost phụ cho random search B1): cấu hình tune không hơn E8.

## 9. Các thử nghiệm dẫn đến cấu hình này
Mọi lựa chọn dựa trên validation; test chỉ để báo cáo.

| | thử nghiệm | kết luận |
|---|---|---|
| E1 | bộ chia sẵn, 10 class | 24,5% Normal bị báo nhầm do train/test lệch phân phối → chia lại từ raw |
| E2 | chia lại, 7 class (gộp 4 class không tách được), 5 model | XGBoost tốt nhất gần như mọi class |
| E3 | chia lại, 10 class | quy về 7 class cho cùng kết quả → báo cáo bản 10 class (không che class khó) |
| E4 | class weight α ∈ {0, 0.25, 0.5, 0.75, 1} | XGB α=0.5 |
| E5 | dò bias ngưỡng từng class trên val | chỉ tái tạo hiệu ứng class weight, overfit val → không dùng |
| E6 | bỏ TTL | gần như không mất điểm → bỏ (tránh shortcut theo máy gửi) |
| E7 | thêm 9 feature tỉ lệ | Shellcode tăng, không hại class nào → giữ; Fuzzers↔Normal không tách thêm được |
| E8/W1 | kết hợp, 3 cách chia | cấu hình cuối; Shellcode tăng ở cả 6 lần đo |
| B1 | random search 20 cấu hình XGB | không hơn E8 ngoài trượt trên đường đánh đổi |
| B3 | torch_mlp α ∈ {0, 0.25, 0.5} (GPU) | α=0.25 |
| C1 / C2 | xoá trùng / gộp Worms | không làm: vi phạm toàn vẹn dữ liệu / che lỗi của model (Worms có recall oracle 0.97) |

Notebook `01_preprocessing.ipynb`, `02_centralized_baseline.ipynb` là lịch sử (bài toán binary, bộ chia sẵn); không dùng số liệu của chúng.
