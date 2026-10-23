# Kiem tra APK bang project

## Chay file AM_CS_NEW.apk o thu muc cha

Mo PowerShell trong android-drift va chay:

```powershell
powershell -ExecutionPolicy Bypass -File .\check_apk.ps1
```

Hoac chay thu cong (Python 3.11+; inspect-apk khong can NumPy):

```powershell
python -m driftlab inspect-apk --apk "..\AM_CS_NEW.apk" --output "runs\apk_first"
Start-Process .\runs\apk_first\apk_report.html
```

Thu muc output phai moi. APK chi duoc doc; khong cai/chay, khong tai len dich vu ngoai.

## Dau ra

- apk_report.json / apk_report.html: SHA256, package, version, SDK, permissions,
  component names va danh sach DEX; status=inspection_only, verdict=null.
- features.json: token dac trung schema android-manifest-v1, khong tu gan nhan.

Bo trich xuat nay doc manifest binary XML hoac text XML. Khong phan tich loi goi
API trong DEX, khong xac minh chu ky, khong tai lap Drebin/APIGraph. Quyen
nhay cam khong du de ket luan malware. Neu gap resource reference thi giu
nguyen @resource:...; chua resolve resources.arsc. Missing flag khong dong
nghia false.

## Du doan bang checkpoint that

```powershell
python -m driftlab predict-apk --apk "..\AM_CS_NEW.apk" --checkpoint "models\real.npz" --model-card "models\real.json" --output "runs\apk_prediction"
```

Can NumPy va checkpoint MLP cua project. Checkpoint phai hoc tu cung schema:
feature_names la permission:android.permission.INTERNET, component:service,
application:debuggable, v.v. Khong chap nhan cot vo danh f_0001. Trong CSV
huan luyen, dung cot f_permission:android.permission.INTERNET; training loader
se giu prefix f_; predictor tu nhan prefix nay khi anh xa dac trung.
Giu nguyen thu tu cot, trong so, mean va scale. Khong doi ten dac trung demo
thanh dac trung that. Checkpoint Drebin/API can extractor tuong ung, khong
duoc ep sang schema manifest nay.

Model card JSON bat buoc cac truong sau (gia tri phai dung voi lan train):

```json
{
  "model_id": "YOUR_REAL_MODEL_ID",
  "feature_schema": "android-manifest-v1",
  "training_data_kind": "real_android",
  "dataset_source": "ACTUAL_DATASET_AND_VERSION",
  "checkpoint_sha256": "ACTUAL_SHA256_OF_NPZ",
  "threshold": 0.5
}
```

Model card la khai bao cua tac gia, khong tu chung minh chat luong mo hinh.
Khong can nhan APK de du doan; can nhan tham chieu de cham dung/sai.
Project chua kem checkpoint du lieu Android that nen chuc nang hien co the
chay ngay la inspect-apk. Khong dung ket qua synthetic de ket luan APK.

## Kiem thu

```powershell
python -m unittest discover -s tests -v
```

Test suy luan dung checkpoint gia lap chi de xac minh tinh toan, khong phai
mo hinh phat hien malware da duoc huan luyen/kiem dinh.
