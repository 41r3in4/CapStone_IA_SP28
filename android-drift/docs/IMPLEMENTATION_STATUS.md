# Implementation status — 2026-10-23

Cập nhật đường dẫn APK: `extract-batch` mặc định đọc `data/apks/`, hỗ trợ thư mục con;
đã tạo thư mục rỗng và thêm `docs/DATA_LAYOUT.md`. Không di chuyển hay xử lý APK có sẵn.

## Đã hoàn thành — công cụ, kiểm thử fixture

| Mục | File chính | Bằng chứng |
|---|---|---|
| B: audit, split manifest, temporal validation | `driftlab/protocol.py`, `validation.py`, `data.py`, `preparation.py`, `__main__.py` | `test_protocol.py`: split disjoint/đủ ID, monitor không tham gia validation, feature tương lai không được chọn; `test_preparation.py`: exact vocabulary fit IDs |
| B: sparse loader và pipeline CSR | `driftlab/data.py`, `protocol.py`, `selection.py`, `experiment.py` | CSR load/shape validation; `test_sparse_run_without_densification` cấm gọi `toarray()` trong toàn run |
| C1: query manifest và fixed replay | `driftlab/queries.py`, `experiment.py` | ID sai tháng/trùng/vượt ceil budget/tháng cuối bị từ chối; hai updater phát lại cùng ID; nhãn monitor đổi không đổi query/prediction |
| C2: EWC độc lập, SVM, matrix, ablation và lock | `driftlab/svm.py`, `suite.py`, `locking.py`, `scripts/run_suite.py` | LinearSVC frozen test; ma trận ablation nhỏ 6 cấu hình; thiếu fixed manifest/lock bị chặn; CLI lock test |
| C3: metric, timing, peak RAM | `metrics.py` giữ logic hiện có; `experiment.py`, `resources.py` thêm accounting | Test metric biên/null và tách nhãn; Windows API đo peak RSS thành công; GPU không dùng |
| C4: aggregation, Wilcoxon/Holm, Friedman permutation, moving-block bootstrap | `driftlab/statistics.py`, `analysis.py` | Test paired mismatch/ít unit, Holm, Friedman, bootstrap tái lập, analysis ít seed trả N/A |
| D: export/predict package | `driftlab/package.py`, `apk.py`, `__main__.py` | Fixture build → train → export → predict; hash/schema/order/preprocessing/architecture mismatch bị từ chối; synthetic giữ inspection_only ngoài fixture API |
| E1: resume | `driftlab/resume.py`, `experiment.py`, `__main__.py` | Chạy liên tục vs resume: weights/predictions/query/RNG/memory/seen/observed bit-exact trong fixture; config mismatch và nhãn chưa query trong state bị từ chối; CLI resume pass |
| E2: môi trường/tài liệu | `requirements-verified.txt`, `configs/real.template.json`, `configs/model-card.synthetic.json`, `README.md`, `docs/STAGES_B_E.md`, `scripts/verify.py`, `VERIFICATION.json` | Dependency thực tế đã cài và test; log kiểm thử và source hashes được lưu |

Kiểm thử: **29/29 pass**, gồm 16 test ban đầu và 13 test bổ sung. Không skip.
`git diff --check`: exit 0. Kiểm tra whitespace bổ sung cho cả file source mới: 0 lỗi.
Lệnh chính thức: `.\.venv\Scripts\python.exe scripts/verify.py`.
Kết quả đo thực tế, thời gian, dependency, hash và log nằm trong `VERIFICATION.json` và `docs/verification-tests.log`.
Không tuyên bố chất lượng phát hiện Android từ test synthetic.

## Chưa hoàn thành — cần đầu vào ngoài

- Chưa chạy bất kỳ lưới nghiên cứu Android thật nào, gồm 3 budget × 5 seed và thống kê trên kết quả thật.
- Chưa có corpus/metadata nguồn nhãn/tháng đã xác minh và protocol thật đã review/khóa.
- Chưa có model Android thật tương thích được đánh giá; chưa đo accuracy/calibration trên APK thật.
- Không đánh giá `AM_CS_NEW.apk`, không cài/chạy APK, không upload, commit hoặc push.
- Chưa benchmark RAM quy mô Drebin thật. CSR tránh densify toàn dataset nhưng weights MLP và vector moment vẫn dense theo chiều feature.
- Feature provenance ngoài thiếu bằng chứng train-only vẫn cần audit thủ công; code không thể tự chứng minh upstream không dùng tương lai.
- SVM là baseline frozen riêng; không hỗ trợ EWC hoặc package MLP cho SVM.
- Resume ở ranh giới tháng, sang thư mục mới; bit-exact đã kiểm tra trên fixture cùng môi trường, không hứa giữa BLAS/OS/version khác. Wall time và peak RSS không bit-exact.
- CADE, pseudo-loss, hierarchical contrastive, online accumulated Fisher và replay cân bằng lớp/thời gian nằm ngoài B → E này; heuristic vẫn giữ tên đúng.

## Bị chặn / lỗi môi trường

- Terminal mặc định trong sandbox không khởi tạo được: `helper_unknown_error: setup refresh had errors`.
  Công việc đã thực hiện bằng terminal ngoài sandbox qua **cơ chế phê duyệt chính thức**; không tự vượt quyền.
- Git ban đầu báo `detected dubious ownership`. Đã dùng `git -c safe.directory=H:/CapStone_IA_SP_28/CapStone_IA_SP_28 diff --check`; không sửa config Git toàn cục.
- SciPy và scikit-learn ban đầu thiếu; đã cài vào `.venv` qua phê duyệt và kiểm thử thành công. Không còn bị chặn dependency.
- Phần nghiên cứu thật bị chặn bởi thiếu dữ liệu/provenance/protocol/model thật đã xác thực, không phải lỗi test công cụ.

## Dependency đã thực sự kiểm thử

Windows x64, CPython 3.14.7; NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1,
joblib 1.6.0, threadpoolctl 3.7.0, narwhals 2.27.1, cloudpickle 3.1.2.
Xem `requirements-verified.txt`; Python/version khác chưa được kiểm thử trong phiên này.
Các file/thay đổi có sẵn của người dùng được giữ; không reset Git, xóa dữ liệu hoặc commit.
