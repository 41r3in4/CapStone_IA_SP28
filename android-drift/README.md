# Android Drift - Project capstone khởi đầu

Project chạy được cho vòng **kiểm thử trước, cập nhật sau** trên dữ liệu đặc trưng theo tháng. Đây là bộ nền thực nghiệm, chưa phải bản hoàn tất toàn bộ đề cương, chưa phải chương trình quét APK và chưa được xác thực trên dữ liệu Android thật.

B → E đã bổ sung công cụ và kiểm thử: xem [hướng dẫn Windows](docs/STAGES_B_E.md), [tiến độ và giới hạn](docs/IMPLEMENTATION_STATUS.md) và [bằng chứng kiểm thử](VERIFICATION.json). Synthetic và dữ liệu thật có cấu hình riêng. Chưa chạy nghiên cứu Android thật.

## 1. Chạy ngay trên Windows

Giải nén ZIP, mở PowerShell tại thư mục chứa README này. Cài Python 3.11 trở lên nếu chưa có, rồi chạy:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-verified.txt
.\.venv\Scripts\python.exe -m driftlab demo-data
.\.venv\Scripts\python.exe -m driftlab run --config configs/demo.json
Start-Process .\runs\demo\report.html
```

Linux/macOS: thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`. Các đường dẫn trong cấu hình tính từ thư mục hiện tại; chạy tại thư mục gốc project. Không cần kích hoạt môi trường. Không chạy lại demo-data trên cùng tệp hoặc ghi đè thư mục run; dùng `--output` mới.

Demo tạo 20 tháng, 180 mẫu/tháng, 24 đặc trưng nhị phân có biến đổi phân phối. **Số liệu demo không được dùng để kết luận về mã độc Android.** Demo giúp kiểm tra chương trình và trình bày cơ chế hoạt động.

## 2. Các lệnh thực nghiệm

```powershell
# B0: không cập nhật
.\.venv\Scripts\python.exe -m driftlab run --updater frozen --selector random --output runs/b0
# Mốc toàn bộ nhãn: không cùng ngân sách với AL
.\.venv\Scripts\python.exe -m driftlab run --updater full_oracle --output runs/full_oracle
# AL ngẫu nhiên + huấn luyện lại trên dữ liệu nhãn tích lũy
.\.venv\Scripts\python.exe -m driftlab run --selector random --updater retrain --output runs/random_retrain
# AL ngẫu nhiên + replay bộ nhớ giới hạn
.\.venv\Scripts\python.exe -m driftlab run --selector random --updater replay --output runs/random_replay
# Hybrid + replay + EWC
.\.venv\Scripts\python.exe -m driftlab run --selector hybrid --updater replay_ewc --output runs/hybrid_ewc
# Chạy bảng so sánh 7 cấu hình
.\.venv\Scripts\python.exe scripts/run_suite.py --output runs/comparison
# Lưới 3 ngân sách, 5 seed (tốn thời gian hơn)
.\.venv\Scripts\python.exe scripts/run_suite.py --output runs/full_grid --full-grid
# Kiểm tra chương trình
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Selector: `random`, `uncertainty`, `conformal`, `distance`, `hybrid`.
Updater: `frozen`, `full_oracle`, `retrain`, `finetune`, `ewc`, `ewc_reg`, `replay`, `replay_reg`, `replay_ewc`, `replay_ewc_reg`.

## 3. Đầu ra và cách đọc

Mỗi run tạo:
- `config.json`: cấu hình chính xác.
- `splits.csv`: ID train, calibration, monitor và future test.
- `monthly_metrics.csv`: F1/FNR/FPR trước cập nhật, hồi quy sau cập nhật, nhãn và thời gian.
- `queries.csv`: các ID thực sự truy vấn; có thể không có nếu không truy vấn lần nào.
- `predictions.csv`: dự đoán trước cập nhật để kiểm tra thứ tự thời gian.
- `summary.json`: AUT, tổng nhãn, phiên bản, hash dữ liệu và giới hạn.
- `model_YYYY-MM.npz` và `state_YYYY-MM.json`: trọng số, preprocessing, RNG, replay buffer, nhãn quan sát và log để resume sang thư mục mới.
- `query_manifest.csv`: run ID, tháng, ID và thứ tự truy vấn; luôn có header kể cả không có truy vấn.
- `data_audit.json`, `preprocessing.json`: audit và trạng thái chỉ fit trên train.
- `report.html`: báo cáo mở offline, không cần server hoặc tài khoản.

Tháng cuối chỉ đánh giá, không truy vấn/cập nhật vì không còn tháng tương lai để đo hiệu quả. Ngân sách truy vấn không tính nhãn ban đầu; summary tách fit, calibration và monitor. Monitor/calibration giữ riêng, không cho vào replay.

## 4. Dữ liệu thật

APK mẫu mặc định đặt trong `data/apks/` (thuộc `data/`). Chạy `.\.venv\Scripts\python.exe -m driftlab extract-batch --output data/extraction-v1` để trích xuất tĩnh. Xem [cấu trúc dữ liệu](docs/DATA_LAYOUT.md). Không có APK được tự tải hoặc tự đưa vào train; nhãn/tháng vẫn cần metadata có nguồn xác minh.

Xem `docs/IMPLEMENTATION.md`. Định dạng CSV: `sample_id,month,label,f_*`. Nhãn 0/1, tháng YYYY-MM, ID duy nhất; mọi tháng phải liên tiếp. Nạp dữ liệu offline vào RAM nên đây không phải triển khai giới hạn toàn bộ RAM. Chuẩn hóa chỉ fit trên train ban đầu. Không tự cân bằng tỷ lệ kiểm thử về 10%; người dùng phải chuẩn bị đúng giao thức và báo cáo tỷ lệ lớp.

Đặc trưng Drebin rất rộng: **không nên xuất ma trận cực lớn sang CSV**. Adapter cũ vẫn có giới hạn dense. Loader mới nhận thư mục CSR gồm `matrix.npz`, `metadata.csv`, `features.json`; không dùng object pickle và không densify toàn ma trận. Feature selection và scale chỉ fit trên train. Xem định dạng trong `docs/STAGES_B_E.md`.

## 5. Đã làm và chưa làm

| Đã có mã chạy | Chưa có / khác đề cương |
|---|---|
| MLP một lớp ẩn; baseline LinearSVC frozen riêng | Hierarchical contrastive, nhãn họ |
| Conformal theo lớp, score xác suất | Chưa tái lập toàn bộ ICE/CCE của Transcendent |
| Khoảng cách centroid/MAD trong latent | Không phải CADE contrastive autoencoder |
| Hybrid credibility + uncertainty + distance + đa dạng | Chưa có pseudo-loss của Chen |
| Uniform reservoir replay giới hạn mẫu | Chưa có replay cân bằng lớp/thời gian hoặc ưu tiên |
| Diagonal empirical Fisher + EWC neo vào mô hình trước | Chưa có accumulated/online Fisher nhiều mốc |
| BCE distillation trên mẫu mã độc teacher phát hiện đúng | Không phải tái lập PCT nguyên bản |
| Metric, AUT, regression; suite, Wilcoxon/Holm, Friedman permutation, block bootstrap | Chưa có kết quả thống kê trên dữ liệu Android thật |
| Báo cáo HTML, thời gian và peak RSS của process | Chưa dùng GPU, chưa xác thực model Android thật |

Không đổi tên các heuristic thành phương pháp gốc trong báo cáo. Không có tuyên bố vượt baseline. Demo vẫn dùng split phân tầng; cấu hình `temporal` và `validation_candidates` kích hoạt validation quá khứ tách riêng. Bộ dữ liệu ngoài phải chứng minh vocabulary upstream chỉ fit trên train.

## 6. Nguồn và thứ tự phát triển

1. [Chen et al., USENIX 2023](https://www.usenix.org/conference/usenixsecurity23/presentation/chen-yizheng), [mã/dữ liệu](https://github.com/wagner-group/active-learning).
2. [TESSERACT](https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury).
3. [Transcendent](https://github.com/s2labres/transcendent-release).
4. [CADE](https://github.com/whyisyoung/CADE).
5. [APIGraph](https://github.com/seclab-fudan/APIGraph).
6. [Regression-aware CL](https://arxiv.org/abs/2507.18313).

Mã trong ZIP được viết riêng cho starter này; không sao chép toàn kho của tác giả. Thứ tự mở rộng: audit dữ liệu → sparse loader + SVM → validation → tái lập selector/mô hình gốc → replay chọn lọc → thí nghiệm thật → kiểm định. Xem `docs/IMPLEMENTATION.md` cho tiêu chí từng bước.
