# UNSW-NB15: baseline tập trung cho Federated Learning

Dự án phát hiện xâm nhập mạng trên bộ dữ liệu UNSW-NB15 (nhãn nhị phân `label`: 0 là bình thường, 1 là tấn công). Gồm tiền xử lý dùng chung (cùng cách với nhóm federated) và huấn luyện tập trung làm mốc so sánh cho federated.

## Cấu trúc thư mục

```
Cloud/
├── README.md
├── 01_preprocessing.ipynb          # tiền xử lý dùng chung cho cả nhóm
├── 02_centralized_baseline.ipynb   # baseline tập trung: tinh chỉnh, chọn ngưỡng, 5 seed, CI 95%
├── CSV Files/
│   ├── Training and Testing Sets/
│   │   ├── UNSW_NB15_training-set.csv   # đầu vào
│   │   └── UNSW_NB15_testing-set.csv    # đầu vào
│   └── ...                         # các file thô UNSW-NB15_1..4.csv, GT, features, ... (không notebook nào dùng)
├── output_preprocess/              # tạo ra bởi 01
│   ├── preprocess.joblib
│   └── data.npz
└── output_baseline/                # tạo ra bởi 02
    ├── summary.csv
    ├── best_params.json
    ├── runs.csv
    ├── per_category.csv
    ├── bootstrap_ci.csv
    ├── bootstrap_pairwise.csv
    └── models/                     # logistic_regression, mlp, xgboost, random_forest (.joblib)
```

## Yêu cầu và cài đặt

- Python 3.9 trở lên
- Thư viện: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `joblib`, `xgboost`, `jupyter`

Chạy trong thư mục `Cloud/`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib xgboost jupyter ipykernel
python -m ipykernel install --prefix .venv --name cloud-venv --display-name "Python (.venv Cloud)"
```

Chỉ cần cài một lần. Trên macOS dùng Python của Homebrew, `pip install` bên ngoài venv sẽ báo lỗi `externally-managed-environment`, nên phải cài trong `.venv` như trên. Mỗi lần mở terminal mới, chạy lại `source .venv/bin/activate` trước khi dùng `jupyter`.

## Cách chạy

Chạy theo thứ tự **01 rồi 02**, vì 02 đọc `output_preprocess/data.npz` do 01 tạo ra. Thư mục làm việc phải là `Cloud/` vì đường dẫn dữ liệu là đường dẫn tương đối.

| Notebook | Cần chạy trước | Thời gian (ước lượng) |
|---|---|---|
| `01_preprocessing.ipynb` | không | dưới 1 phút |
| `02_centralized_baseline.ipynb` | `01` | khoảng 12 phút |

**Jupyter:**

```bash
source .venv/bin/activate
jupyter notebook
```

Trình duyệt sẽ mở. Mở notebook, chọn kernel **Python (.venv Cloud)**, rồi chọn **Kernel → Restart & Run All**. Tắt server bằng `Ctrl+C` trong terminal.

**VS Code:** cài extension **Jupyter** (Microsoft), mở notebook, bấm **Select Kernel** (góc phải trên) và chọn môi trường `.venv`, rồi bấm **Run All**.

**Dòng lệnh:**

```bash
jupyter nbconvert --to notebook --execute --inplace 01_preprocessing.ipynb --ExecutePreprocessor.kernel_name=cloud-venv
jupyter nbconvert --to notebook --execute --inplace 02_centralized_baseline.ipynb --ExecutePreprocessor.kernel_name=cloud-venv --ExecutePreprocessor.timeout=3600
```

## 01: tiền xử lý dùng chung

Tính thống kê theo kiểu federated: mỗi client gửi `(n, mean, M2)` và tập category, server gộp bằng công thức Chan, kết quả bằng đúng bản tập trung (sai khác khoảng 1e-15). Các bước: điền thiếu (`ffill`, `bfill`), bỏ `id` và `attack_cat`, đổi `ackdat` thành `ackdata`, `log1p` rồi chuẩn hóa 39 cột số, One-Hot cho `proto`, `service`, `state` (bỏ giá trị đầu). Không dùng PCA. Kết quả: 191 đặc trưng, train 175.341 dòng (68,06% tấn công), test 82.332 dòng (55,06% tấn công).

Đầu ra trong `output_preprocess/`, gửi thư mục này cho cả nhóm để dữ liệu đồng bộ:

| File | Mô tả |
|---|---|
| `preprocess.joblib` | `mean`, `std`, `vocab`, `num_cols`, `cat_cols`, `features` (thứ tự 191 cột) |
| `data.npz` | `X_train`, `y_train`, `attack_cat_train`, `X_test`, `y_test`, `attack_cat_test`, `features` (khoảng 10 MB, đọc được với `allow_pickle=False`) |

Cuối notebook có hướng dẫn dùng `data.npz`, xử lý dữ liệu thô mới bằng `preprocess.joblib`, và chạy theo kiểu federated thật.

## 02: baseline tập trung

**Quy trình:** tách validation theo nhóm trùng lặp (khoảng 80/20, phân tầng theo `attack_cat`, các dòng giống hệt nhau luôn nằm cùng một phía), tinh chỉnh tham số bằng random search trên validation (tiêu chí ROC-AUC), chọn ngưỡng quyết định trên validation, huấn luyện lại với 5 seed, chấm một lần trên tập test chính thức. LR và MLP chỉ dùng SGD không momentum để tương thích FedAvg.

**Ba điểm hoạt động (ngưỡng chọn trên validation):**

| Điểm | Cách chọn |
|---|---|
| `τ=0,5` | Ngưỡng mặc định |
| `Youden` | Cực đại hóa TPR − FPR, không phụ thuộc tỷ lệ lớp |
| `FPR≤10%` | Recall cao nhất khi FPR trên validation không vượt 10% |

### Kết quả chính

Cấu hình đã tinh chỉnh, toàn bộ tập test, trung bình ± độ lệch chuẩn qua 5 seed. Recall, Precision, F1 là của lớp Tấn công; FPR là tỷ lệ luồng bình thường bị báo nhầm. XGBoost có std bằng 0 vì kết quả không phụ thuộc seed khi `subsample=1` và `colsample_bytree=1`. Toàn bộ số liệu nằm trong `output_baseline/summary.csv`.

**Ngưỡng Youden (khuyến nghị dùng để so sánh):**

| Mô hình | Recall | Precision | FPR | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0,9623 ± 0,0038 | 0,8781 ± 0,0077 | 0,1639 ± 0,0124 | **0,9182** ± 0,0025 | 0,9056 | **0,9821** |
| MLP | 0,9557 ± 0,0034 | 0,8826 ± 0,0088 | 0,1559 ± 0,0135 | 0,9177 ± 0,0046 | 0,9055 | 0,9801 |
| XGBoost | 0,9541 | 0,8792 | 0,1605 | 0,9152 | 0,9026 | 0,9812 |
| Logistic Regression | 0,9274 ± 0,0084 | 0,8504 ± 0,0104 | 0,2001 ± 0,0179 | 0,8872 ± 0,0021 | 0,8701 | 0,9630 |

**Ngưỡng τ=0,5:**

| Mô hình | Recall | Precision | FPR | F1 | Accuracy |
|---|---|---|---|---|---|
| MLP | 0,9775 | 0,8253 | 0,2539 | 0,8949 | 0,8735 |
| Random Forest | 0,9882 | 0,8151 | 0,2747 | 0,8933 | 0,8700 |
| XGBoost | 0,9784 | 0,8185 | 0,2658 | 0,8914 | 0,8687 |
| Logistic Regression | 0,9805 | 0,7751 | 0,3497 | 0,8656 | 0,8321 |

**Ngưỡng FPR≤10%:**

| Mô hình | Recall | Precision | FPR | F1 |
|---|---|---|---|---|
| MLP | 0,9794 | 0,8181 | 0,2667 | 0,8915 |
| Random Forest | 0,9895 | 0,8107 | 0,2830 | 0,8913 |
| XGBoost | 0,9831 | 0,8043 | 0,2931 | 0,8847 |
| Logistic Regression | 0,9527 | 0,8147 | 0,2655 | 0,8783 |

Ngưỡng 0,5 cho FPR rất cao (25–35%) vì train có 68% tấn công còn test chỉ 55%. Ngưỡng `FPR≤10%` giữ được FPR dưới 10% trên validation nhưng lên tới 27–29% trên test, cho thấy phân phối luồng bình thường giữa train và test khác nhau. Ngưỡng Youden cho F1 và FPR tốt nhất.

### Khoảng tin cậy 95% (bootstrap theo nhóm trùng lặp, seed 0, ngưỡng Youden)

| Mô hình | ROC-AUC | F1 | Recall | FPR |
|---|---|---|---|---|
| Random Forest | 0,9822 [0,9813; 0,9830] | 0,9178 [0,9147; 0,9208] | 0,9639 [0,9614; 0,9662] | 0,1672 [0,1626; 0,1718] |
| MLP | 0,9809 [0,9799; 0,9818] | 0,9206 [0,9173; 0,9236] | 0,9573 [0,9546; 0,9596] | 0,1501 [0,1456; 0,1544] |
| XGBoost | 0,9812 [0,9802; 0,9821] | 0,9152 [0,9120; 0,9181] | 0,9541 [0,9515; 0,9565] | 0,1605 [0,1565; 0,1651] |
| Logistic Regression | 0,9634 [0,9617; 0,9647] | 0,8893 [0,8856; 0,8929] | 0,9223 [0,9187; 0,9255] | 0,1862 [0,1817; 0,1909] |

So sánh ghép cặp (`bootstrap_pairwise.csv`): Logistic Regression kém hơn rõ rệt so với ba mô hình còn lại. Giữa MLP, XGBoost và Random Forest, chênh lệch ROC-AUC chỉ khoảng 0,001 (MLP và XGBoost không khác nhau có ý nghĩa), nên ba mô hình này gần như tương đương.

### Recall theo loại tấn công (ngưỡng Youden, trung bình 5 seed)

| Loại | LR | MLP | XGBoost | RF |
|---|---|---|---|---|
| Generic | 0,998 | 0,999 | 1,000 | 1,000 |
| Backdoor | 0,983 | 0,995 | 0,998 | 0,998 |
| Reconnaissance | 0,971 | 0,992 | 0,997 | 0,996 |
| DoS | 0,969 | 0,990 | 0,995 | 0,995 |
| Exploits | 0,933 | 0,980 | 0,987 | 0,990 |
| Shellcode | 0,937 | 0,977 | 0,974 | 0,982 |
| Worms | 0,955 | 0,995 | 1,000 | 0,977 |
| Analysis | 0,915 | 0,994 | 0,934 | 0,958 |
| Fuzzers | 0,639 | 0,722 | 0,697 | 0,749 |

Fuzzers là loại khó phát hiện nhất ở mọi mô hình.

### Kết quả phụ

- **Phần test chưa thấy** (bỏ khoảng 10% dòng test có bản sao y hệt trong train, ngưỡng Youden): F1 giảm khoảng 0,01, ví dụ RF 0,9085, MLP 0,9075, XGBoost 0,9040, LR 0,8714.
- **Mặc định so với tinh chỉnh** (ROC-AUC test): LR 0,9526 → 0,9630, MLP 0,9778 → 0,9801, RF 0,9819 → 0,9821, XGBoost 0,9832 → 0,9812. Tinh chỉnh giúp rõ nhất với LR; với XGBoost thì giảm nhẹ trên test dù tốt hơn trên validation.
- **Chi phí** (trung bình mỗi lần train): LR 3,2 s, XGBoost 3,4 s, MLP 15,6 s, RF 18,9 s. Kích thước: LR 192 tham số, MLP 34.945 tham số, XGBoost 500 cây, RF 300 cây (file khoảng 100 MB).

### Đầu ra trong `output_baseline/`

| File | Mô tả |
|---|---|
| `summary.csv` | Bảng kết quả chính ở trên (mô hình × điểm hoạt động, trung bình và std) |
| `best_params.json` | Siêu tham số tốt nhất cho 4 mô hình |
| `runs.csv` | Kết quả từng lần chạy (mô hình × cấu hình × seed × điểm hoạt động × tập test) |
| `per_category.csv` | Recall theo loại tấn công |
| `bootstrap_ci.csv`, `bootstrap_pairwise.csv` | Khoảng tin cậy 95% và so sánh ghép cặp |
| `models/*.joblib` | Mô hình đã tinh chỉnh ở seed 0 |

Dùng lại mô hình đã lưu trên dữ liệu đã xử lý:

```python
import joblib, numpy as np
d = np.load('output_preprocess/data.npz', allow_pickle=False)
model = joblib.load('output_baseline/models/xgboost.joblib')
prob = model.predict_proba(d['X_test'])[:, 1]
```

## Bàn giao cho nhóm federated

Để so sánh công bằng với baseline này, phần federated nên dùng:
- cùng tiền xử lý: `output_preprocess/preprocess.joblib` hoặc `data.npz`;
- cùng siêu tham số: `output_baseline/best_params.json` (LR và MLP là SGD không momentum, dùng được với FedAvg);
- cùng cách chấm: tập test chính thức, báo cáo ở ngưỡng `Youden` và `τ=0,5`, kèm ROC-AUC.

## Lưu ý

- UNSW-NB15 train và test cùng bắt nguồn từ các file thô `UNSW-NB15_1..4.csv`, và khoảng 10% dòng test có bản sao y hệt trong train, nên điểm số có thể hơi lạc quan.
- Tập test có hai giá trị `state` (`ACC`, `CLO`) không có trong train, được mã hóa thành toàn 0.
