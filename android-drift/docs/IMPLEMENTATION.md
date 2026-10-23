# Chi tiết thực hiện với project

Cập nhật B → E: xem [STAGES_B_E.md](STAGES_B_E.md), [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) và `../VERIFICATION.json`. Các mục sau giữ bối cảnh starter; chức năng đã bổ sung được ghi rõ bên dưới.

## A. Đọc mã theo thứ tự

1. `data.py`: định dạng, kiểm tra dữ liệu, 12 tháng đầu và các tập giữ riêng.
2. `model.py`: forward, BCE, backprop, EWC và Fisher. Đây là MLP NumPy để dễ kiểm tra; không phải mô hình tương phản phân cấp.
3. `selection.py`: bộ chọn chỉ nhận X, xác suất và nhãn tham chiếu đã biết; không nhận y của tháng hiện tại.
4. `experiment.py`: vòng tháng, oracle, bộ nhớ, updater, checkpoint và log.
5. `metrics.py`, `report.py`: chỉ số và báo cáo. Không dùng số liệu sau cập nhật làm F1 tháng hiện tại.

## B. Các bước chạy và mục đích

### Bước 1 - Chạy demo và unit test
Chạy lệnh README. Xem splits.csv để xác nhận ba tập ban đầu không trùng. Mở predictions.csv và queries.csv để phân biệt dự đoán với truy vấn. Đầu ra đạt: test pass, report mở được, không có lỗi số chiều hay nhãn.

### Bước 2 - So sánh không cập nhật và toàn bộ nhãn
Chạy frozen và full_oracle. Xem monthly_metrics và summary. Mục đích: hiểu mốc không thích nghi và mốc nhiều nhãn. Không lấy số mô phỏng làm minh chứng drift Android; không đòi hỏi đường cong phải giảm hoặc phương pháp mới phải thắng.

### Bước 3 - Tách hiệu quả chọn mẫu
Giữ updater=replay; chạy random, uncertainty, conformal, distance, hybrid. Giữ seed, beta, dữ liệu và mô hình ban đầu. Mục đích: đánh giá toàn vòng AL với từng selector. Để cô lập updater, dùng `--query-manifest` hoặc suite `--mode updaters` với mapping manifest theo budget/seed. Chế độ AL khép kín và fixed-query replay được ghi riêng trong summary.

### Bước 4 - Tách hiệu quả cập nhật
Thử finetune, replay, replay_ewc và replay_ewc_reg. Chú ý selector phụ thuộc mô hình nên chuỗi mẫu có thể khác; gọi đây là so sánh end-to-end nếu chưa cố định truy vấn. Đầu ra: bảng AUT, tổng nhãn và thời gian từng phương pháp.

### Bước 5 - Audit dữ liệu nghiên cứu
Tải D1/D2 từ nguồn tác giả. Xem tên tệp và xác nhận zero là benign; nhãn họ khác zero không có nghĩa mọi tập dữ liệu đều dùng quy ước này. Kiểm tra các tháng, số chiều và nguồn thời gian. Dùng hash APK thật nếu có; ID source-row chỉ để truy vết trong adapter.

### Bước 6 - Nhập NPZ quy mô nhỏ
Adapter chỉ lấy file YYYY-MM_selected.npz, loại file gộp để tránh nhập một mẫu hai lần. Các trường mong đợi là X_train/y_train. Lệnh:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-import.txt
.\.venv\Scripts\python.exe scripts/import_npz.py --input data/gen_apigraph_drebin --output data/real_small.csv --deduplicate
.\.venv\Scripts\python.exe -m driftlab run --data data/real_small.csv --updater frozen --output runs/real_b0
```

Nếu NPZ chứa object pickle, mặc định importer sẽ từ chối. Chỉ thêm `--trusted-pickle` khi đã xác minh nguồn tệp gốc; không dùng cho tệp lạ. Loader CSR mới nhận numeric SciPy NPZ và metadata riêng, không cần bật pickle. Giới hạn phần tử của adapter CSV vẫn giữ nguyên; xem `STAGES_B_E.md` để dùng sparse trực tiếp.

`--deduplicate` giữ vector xuất hiện đầu tiên theo tháng và ghi dropped.csv. Cần thảo luận chính sách này trong protocol: vector giống nhau có thể là ứng dụng khác; xung đột nhãn cần được phân tích, không âm thầm chọn nhãn thuận lợi.

### Bước 7 - Chuẩn hóa giao thức thật
Hiện tại split 70/15/15 theo lớp trong 12 tháng đầu để demo. Bổ sung validation theo thời gian trong giai đoạn quá khứ, chọn siêu tham số và khóa chúng trước test. Kiểm tra bộ chọn đặc trưng của dữ liệu tác giả có dùng tương lai hay không. Chỉ dùng dữ liệu train để fit vocabulary/feature selection. Ghi tỷ lệ mã độc từng tháng và quy tắc lấy mẫu kiểm thử.

### Bước 8 - Nâng cấp về đúng đề cương
Thay latent heuristic bằng CADE được tái lập; thay uncertainty trong hybrid bằng pseudo-loss tham chiếu; thêm hierarchical contrastive khi đủ nhãn họ. Thay uniform reservoir bằng replay chọn lọc có quy tắc lớp/thời gian. Mỗi thay đổi phải có baseline và test riêng, giữ bản starter để đối chiếu.

### Bước 9 - D3 và nhãn
Đăng ký quyền AndroZoo, lưu hash/timestamp/ngày quét/số động cơ phát hiện. Mặc định đề cương: 0 phát hiện benign, >=4 malware, ở giữa loại; phân tích ngưỡng 1/4/10. Đây là tạo nhãn tham chiếu, không phải nhãn tuyệt đối. Project chưa tự gọi API hoặc trích xuất APK. Chỉ xử lý APK tĩnh trong môi trường cô lập và theo điều khoản nguồn.

### Bước 10 - Thực nghiệm và báo cáo
Chạy 3 beta × 5 seed sau khi khóa protocol. Đo peak RAM/GPU bằng công cụ phù hợp nếu báo cáo hiệu quả tài nguyên. Thêm kiểm định theo tháng có cân nhắc phụ thuộc thời gian và block bootstrap AUT. Ghi kết quả phủ định. Demo dashboard hiển thị kết quả lưu sẵn, không nói đang quét APK thời gian thực.

## C. Kiểm soát khoa học
- Label budget: chỉ oracle cấp nhãn mẫu đã chọn; các tập ban đầu được tính riêng.
- Replay cap: chỉ giới hạn buffer học, không giới hạn RAM giữ dữ liệu offline, nhãn evaluator hoặc file checkpoint.
- Calibration cố định từ quá khứ có thể lỗi thời. Conformal score không bảo đảm validity khi exchangeability bị vi phạm.
- Lưu cả tử/mẫu số hồi quy; trường hợp không đủ mẫu trả None/N/A.
- Final month không cập nhật; nếu đổi giao thức phải ghi rõ cách tính số nhãn.
- Snapshot `state_YYYY-MM.json` đi cùng `model_YYYY-MM.npz` lưu RNG, lịch sử, nhãn quan sát và reservoir; resume sang thư mục mới có kiểm tra dataset/schema/config.
