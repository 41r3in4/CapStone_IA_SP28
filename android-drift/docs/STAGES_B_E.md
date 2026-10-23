# B → E: giao thức, thực nghiệm và bàn giao

Chạy PowerShell tại `H:\CapStone_IA_SP_28\CapStone_IA_SP_28\android-drift`.
Mỗi output phải mới. Các lệnh sau là công cụ/template; không có kết quả nghiên cứu Android thật đi kèm.

## Môi trường đã kiểm thử

Windows x64, CPython 3.14.7, dependency chính xác trong `requirements-verified.txt`.
Phiên bản Python khác chưa được kiểm thử trong phiên này. SciPy cần cho protocol/CSR/thống kê;
scikit-learn cần cho LinearSVC. APK inspection vẫn chỉ dùng standard library.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-verified.txt
.\.venv\Scripts\python.exe scripts/verify.py
```

`verify.py` chạy toàn bộ unittest, kiểm tra whitespace và lưu `VERIFICATION.json` cùng log
`docs/verification-tests.log`. Git dùng safe.directory chỉ trong lệnh kiểm tra, không sửa config toàn cục.

## B — dữ liệu, manifest và validation

CSV: `sample_id,month,label,f_*`. CSR là thư mục chứa:

- `matrix.npz`: numeric SciPy CSR lưu bằng `scipy.sparse.save_npz`; không chứa object pickle.
- `metadata.csv`: `sample_id,month,label`, cùng thứ tự hàng với matrix.
- `features.json`: JSON array tên feature duy nhất có tiền tố `f_`, cùng thứ tự cột matrix.
- `provenance.json` nếu có: nguồn feature, hash dataset, các `vocabulary_fit_ids`.

Loader không tự suy diễn nhãn, thời gian hay nguồn từ tên file/timestamp.
ID phải duy nhất; vector trùng được báo cáo cả xung đột nhãn và giao nhau giữa các split.
Vector trùng không tự chứng minh cùng APK. Audit không tự xóa mẫu hoặc sửa nhãn.

```powershell
.\.venv\Scripts\python.exe -m driftlab audit-data --data data/real/dataset.csv --output data/real/audit-v1 --train-through 2022-08 --validation-through 2022-09 --calibration-through 2022-10 --monitor-through 2022-12
```

Các mốc trong `configs/real.template.json` chỉ là ví dụ. Cần thay bằng khoảng dữ liệu thật đã xác minh.
Thứ tự bắt buộc: train → validation → calibration → regression monitor → future test.
Mỗi phần quá khứ cần cả hai lớp, tất cả các tháng phải liên tục.
`splits.csv` phủ mỗi ID đúng một lần. Monitor không được dùng để chọn hyperparameter,
fit vocabulary/preprocessing, train, query hoặc replay.

Stage A `build-dataset --train-through` phải khớp tháng cuối train thực sự,
không phải cuối toàn bộ quá khứ. Có thể dùng `--fit-ids train-ids.json` (JSON array)
để fit vocabulary trên tập ID chính xác. Builder ghi `vocabulary_fit_ids`.
Temporal runner từ chối provenance có vocabulary fit ID ngoài train.
Đặc trưng bên ngoài thiếu provenance được ghi rõ là chưa chứng minh không có leakage upstream;
train-only selection không thể khắc phục một vocabulary vốn đã fit trên tương lai.

`validation_candidates` chỉ nhận `hidden,epochs,learning_rate,batch_size`.
Mỗi candidate fit trên train, chọn F1 cao nhất trên validation; hòa thì chọn candidate đầu.
Validation, calibration và monitor không được nhập lại vào train.
`validation.json` ghi candidate, metric, lựa chọn và preprocessing.
`max_features` chọn feature theo variance train, thứ tự tie ổn định; feature constant trên train bị loại.
Dense được center/scale; CSR chỉ scale, giữ mean=0 để không densify.
MLP vẫn có trọng số dense theo số feature × hidden, vì thế CSR không bảo đảm tổng RAM nhỏ.

## C — so sánh công bằng

```powershell
# AL khép kín, lưu manifest ID theo thứ tự từng tháng
.\.venv\Scripts\python.exe -m driftlab run --config configs/demo.json --output runs/al-source
# Phát lại cùng ID cho updater khác; không chạy selector để chọn lại
.\.venv\Scripts\python.exe -m driftlab run --config configs/demo.json --updater ewc --query-manifest runs/al-source/query_manifest.csv --output runs/ewc-fixed
# Các selector, updater replay cố định
.\.venv\Scripts\python.exe scripts/run_suite.py --mode selectors --output runs/selectors-small
# 3 budget × 5 seed; chỉ chạy khi có dữ liệu và protocol phù hợp
.\.venv\Scripts\python.exe scripts/run_suite.py --config configs/real.template.json --mode closed_loop --full-grid --output runs/real-grid
# SVM frozen riêng, dùng temporal config
.\.venv\Scripts\python.exe -m driftlab svm-run --config configs/real.template.json
```

Query manifest chỉ chứa `run_id,month,sample_id,query_order` (order bắt đầu 0 mỗi tháng),
một run mỗi manifest. ID sai tháng, trùng hoặc vượt `ceil(beta*n)` bị từ chối trước train.
Không truy vấn/cập nhật tháng cuối. Có thể replay ít hơn budget đã cấp; số thực tế được ghi lại.
Full-oracle truy vấn toàn bộ tháng đủ điều kiện, là mốc nhiều nhãn và không nhận fixed manifest.
Frozen không cập nhật. Oracle chỉ tiết lộ nhãn đã chọn; trainer chỉ dùng nhãn train ban đầu và nhãn đã tiết lộ.

Suite `--mode`: `baselines`, `selectors`, `updaters`, `closed_loop`, `ablation`.
`updaters` yêu cầu `--query-manifests manifests.json`, mapping như
`{"0.01:0":"runs/source-b001-s0/query_manifest.csv"}`; full grid cần đủ 15 manifest tương ứng.
`ablation` gồm bỏ conformal, diversity, replay, EWC hoặc regression loss từng phần.
`ewc` và `ewc_reg` không replay; `replay_reg` bỏ EWC.
Các weight còn lại được chuẩn hóa trong ablation bỏ conformal; đây là lựa chọn protocol, không phải phương pháp gốc.
Distance là centroid/MAD heuristic, hybrid dùng uncertainty; không gọi chúng là CADE/pseudo-loss.
LinearSVC là baseline frozen riêng, không được trộn với EWC MLP hoặc xuất package APK MLP.

Trước suite thật, xem audit rồi khóa protocol bằng khai báo rõ:

```powershell
.\.venv\Scripts\python.exe -m driftlab lock-protocol --config configs/real.template.json --output data/real/protocol_lock.json --audit-reviewed --locked-before-test
```

Lock ghi hash dataset, config và source; suite từ chối lock sai hoặc code đổi sau lock.
Các cờ review/thời điểm là khai báo của người chạy, không phải bằng chứng tự động.
Demo được đánh dấu synthetic; không tự đánh dấu dữ liệu thật thành synthetic để bỏ điều kiện.

Metric lưu TP/FP/TN/FN, F1/FNR/FPR trước cập nhật, regression numerator/denominator.
Mẫu số 0 trả null/N/A. Nhãn fit/validation/calibration/monitor/query được đếm riêng.
Thời gian train/selection/update/evaluation/total được ghi; peak RAM là high-water RSS
toàn process (có thể gồm run trước trong suite), không phải RAM riêng của buffer.
Không sử dụng GPU và không báo GPU memory giả.

```powershell
.\.venv\Scripts\python.exe -m driftlab analyze-runs --comparison runs/real-grid/comparison.csv --output runs/real-grid/statistics.json
# Cần >=4 tháng, chỉ định block length có lý do theo dữ liệu
.\.venv\Scripts\python.exe -m driftlab analyze-runs --comparison runs/real-grid/comparison.csv --output runs/real-grid/statistics-block.json --block-length 3
```

Aggregation theo method/budget/seed: mean, sample std và Student-t CI (>=2 seed).
Kiểm định dùng seed AUT ghép cặp trong từng budget; không coi tháng là mẫu độc lập.
Wilcoxon/Holm và Friedman rank với within-block permutation được hỗ trợ khi >=5 paired units.
Thiếu cặp hoặc AUT bị từ chối; ít đơn vị trả N/A. Five seeds có power thấp.
Seed replication chỉ đánh giá randomness trên corpus cố định, không chứng minh tổng quát sang corpus mới.
Block bootstrap lấy block tháng liên tiếp, giữ thứ tự nội bộ; yêu cầu stationarity xấp xỉ,
có thể không phù hợp với drift mạnh. Cần công bố block length, assumptions và phân tích độ nhạy.

## D — model package

```powershell
.\.venv\Scripts\python.exe -m driftlab export-model --run runs/completed-manifest-run --output models/package-v1 --declaration configs/model-card.synthetic.json
.\.venv\Scripts\python.exe -m driftlab predict-package --apk path/to/sample.apk --package models/package-v1 --output runs/package-inspection
# Hợp đồng cũ vẫn dùng được cho package thật tương thích
.\.venv\Scripts\python.exe -m driftlab predict-apk --apk path/to/sample.apk --checkpoint models/package-v1/checkpoint.npz --model-card models/package-v1/model_card.json --output runs/legacy-prediction
```

Exporter chỉ nhận run hoàn thành và vocabulary manifest hợp lệ, không nhận feature demo `f_0000`.
Package gồm checkpoint, schema/vocabulary/architecture, preprocessing, config, provenance,
audit, split manifest và model card. Card giữ hash từng artifact; thứ tự feature, scale/mean,
shape kiến trúc và mapping nhãn được kiểm tra. Hash xác nhận tính nhất quán, không xác thực tác giả.
Source hash là mã lúc train, extractor version là `android-manifest-v1`.
Card ghi nguồn dữ liệu, loại dữ liệu, train period, tháng cập nhật cuối, threshold, mapping nhãn và giới hạn.
Card là khai báo provenance, không chứng minh chất lượng. Score chưa được kiểm chứng calibration.

Package synthetic mặc định trả `inspection_only`, không có verdict APK thật.
API `predict_package(..., fixture=True)` chỉ dùng trong test được đánh dấu synthetic, trả
`fixture_prediction` và verdict null. Không có cờ CLI để biến fixture thành verdict thật.
Không cần nhãn APK mới để predict; nhãn chỉ dùng khi chấm điểm.

## E — resume và bằng chứng

```powershell
.\.venv\Scripts\python.exe -m driftlab run --config configs/demo.json --stop-after-month 2021-01 --output runs/partial
.\.venv\Scripts\python.exe -m driftlab run --config configs/demo.json --resume-from runs/partial/state_2021-01.json --output runs/resumed
```

Snapshot JSON lưu RNG init/selector/train/memory/Fisher, buffer và reservoir seen,
nhãn đã quan sát, completed month, logs; NPZ lưu weights/schema/preprocessing.
EWC hiện neo lại mỗi update: Fisher được tính lại từ model và buffer đã khôi phục,
không cần lưu accumulated Fisher vì thuật toán này không dùng nó.
Resume yêu cầu dataset/schema/config giống nhau, hash checkpoint đúng, không đổi query manifest;
output luôn là thư mục mới, copy log đã có rồi chỉ chạy tháng tiếp theo, không nhân đôi.
Không dùng pickle. Giữ cả JSON và NPZ cùng thư mục. Snapshot hoàn tất theo tháng;
không resume giữa minibatch. Snapshot mới lỗi ghi có thể dùng snapshot tháng trước còn nguyên.
Khuyến nghị giữ nguyên code/dependency đã kiểm thử khi resume.
Test fixture đã so sánh bit-exact weights/query/prediction/RNG/buffer; không hứa bit-exact
giữa OS/BLAS/dependency khác. Thời gian thực thi và peak RSS không phải đại lượng tái lập chính xác.
`total_seconds` là wall time invocation hiện tại; các monthly timing cũ được giữ khi resume.

Chưa chạy: lưới dữ liệu Android thật, đánh giá accuracy thật, thống kê kết quả thật,
calibration score hay đánh giá model trên APK người dùng. Cần corpus/metadata được cấp quyền,
nguồn nhãn/thời gian xác minh, upstream feature provenance và protocol đã khóa.
