# Thư mục dữ liệu mặc định

Chạy lệnh tại thư mục `android-drift`. Đặt APK mẫu trong `data/apks/`;
có thể chia thư mục con tùy ý. `extract-batch` quét đệ quy các file `.apk`.
Tên thư mục không được dùng để tự gán nhãn hoặc tháng quan sát.

```text
data/
  apks/                 APK đầu vào do người dùng cung cấp
  metadata.csv          SHA256, tháng quan sát, nhãn và nguồn đã xác minh
  extraction-v1/        Kết quả trích xuất tĩnh
  dataset-v1/           Dataset đặc trưng và provenance
```

```powershell
.\.venv\Scripts\python.exe -m driftlab extract-batch --output data/extraction-v1
```

Lệnh trên mặc định đọc `data/apks`; có thể dùng `--input` để đổi đường dẫn.
Output phải là thư mục mới. Không tự tải APK, không tự gán nhãn, không cài/chạy APK.
Chỉ đưa APK vào thư mục chưa kích hoạt extraction hay training; phải chạy lệnh tương ứng.
Chưa có APK mẫu được cung cấp sẵn. File `AM_CS_NEW.apk` ngoài project không được tự di chuyển hay xử lý.

Sau extraction, chuẩn bị `data/metadata.csv` có các cột
`sample_id,month,label,source`. `sample_id` là SHA256 từ extraction;
label 0/1 và tháng YYYY-MM phải có căn cứ từ nguồn dữ liệu.
Xem [STAGE_A.md](STAGE_A.md) và [STAGES_B_E.md](STAGES_B_E.md) để build, audit và train.

`data/` được Git ignore để tránh đưa APK/dataset vào repository.
Thư mục có thể được tạo trên checkout mới bằng `New-Item -ItemType Directory -Force data/apks`.
