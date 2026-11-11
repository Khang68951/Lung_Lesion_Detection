# Lung Lesion V75 — Web inference local

English version: [README_EN.md](README_EN.md)

Ứng dụng local nhận một ảnh phổi heo, phân vùng các thùy phổi, khoanh vùng tổn
thương và tính:

```text
tỷ lệ tổn thương (%) = số pixel tổn thương trong phổi / số pixel phổi × 100
```

Toàn bộ source và model cần cho inference đã nằm trong folder này. Không cần
chép `Dataset`, `Legacy_Project` hoặc các project nghiên cứu khác.

> Đây là công cụ nghiên cứu, không phải thiết bị chẩn đoán lâm sàng.

## 1. Yêu cầu máy mới

- Linux 64-bit hoặc Windows 11 chạy WSL2 Ubuntu. Linux/WSL2 được khuyến nghị.
- Python 3.12.
- NVIDIA GPU hỗ trợ CUDA và driver NVIDIA đang hoạt động.
- Khoảng 11 GB để chứa project, cộng thêm dung lượng cho Python/PyTorch.
- Internet trong bước cài thư viện. Khi cài xong, inference không cần tải thêm
  model.

Kiểm tra GPU trước:

```bash
nvidia-smi
```

Nếu lệnh này không nhận GPU thì không nên tiếp tục. Ảnh mới chưa có cache hiện
yêu cầu CUDA; pipeline chưa hỗ trợ CPU-only đầy đủ.

## 2. Những gì cần sao chép

Sao chép nguyên folder và giữ đúng tên:

```text
Unsupervised_Lesion_Workflow/
├── artifacts/                 # Checkpoint V40, V64, V67, V74 và model cache
├── models/
│   ├── sam2.1_hiera_tiny.pt
│   ├── unmarked_lung_unet_best.pt
│   └── dinov2_with_registers_small/
├── src/                       # Python source
├── vendor/sam2/               # Source SAM2 đã pin
├── web_app.py
├── requirements-inference.txt
└── README.md
```

Nên nén bằng `tar` hoặc `zip` có bảo toàn symlink. Sau khi giải nén, không đổi
tên `Unsupervised_Lesion_Workflow` vì một số checkpoint lưu đường dẫn tương đối
theo tên này.

Không sao chép virtual environment từ máy cũ; hãy cài môi trường mới theo phần
tiếp theo.

## 3. Cài đặt từ đầu

Các lệnh dưới đây chạy trong terminal Linux/WSL2, tại thư mục cha đang chứa
`Unsupervised_Lesion_Workflow`.

### 3.1 Cài Python và công cụ hệ thống

Trên Ubuntu/WSL2:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip libgl1 libglib2.0-0
```

### 3.2 Tạo môi trường Python trên máy mới

```bash
cd Unsupervised_Lesion_Workflow
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3.3 Cài PyTorch CUDA

Bản đã kiểm chứng của project là PyTorch `2.5.1+cu121`:

```bash
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Nếu driver/GPU của máy mới cần một bản CUDA khác, chọn lệnh tương ứng tại trang
cài đặt chính thức của PyTorch, nhưng nên kiểm thử lại trước khi dùng lâu dài.

### 3.4 Cài thư viện inference

```bash
python -m pip install -r requirements-inference.txt
```

Không cần chạy `pip install` cho `vendor/sam2`; workflow tự nạp source SAM2 đã
đóng gói.

## 4. Kiểm tra cài đặt

Đảm bảo terminal vẫn đang ở `Unsupervised_Lesion_Workflow` và `.venv` đang được
activate:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Không nhận được NVIDIA GPU/CUDA")
print("GPU:", torch.cuda.get_device_name(0))
PY
```

Kết quả bắt buộc phải có `GPU available: True`.

Kiểm tra model nội bộ mà không chạy dự đoán:

```bash
PYTHONPATH=src python - <<'PY'
from unsupervised_lesion.color_region.student import _dino_provenance
from unsupervised_lesion.io import LUNG_UNET_CHECKPOINT, SAM2_CHECKPOINT

_, dino = _dino_provenance()
assert SAM2_CHECKPOINT.is_file()
assert LUNG_UNET_CHECKPOINT.is_file()
assert dino.is_dir()
print("Model assets: OK")
PY
```

## 5. Quick start — chạy web

Mỗi lần mở terminal mới:

```bash
cd Unsupervised_Lesion_Workflow
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn web_app:app \
  --host 127.0.0.1 --port 8785
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8785
```

Chọn ảnh, bấm **Phân tích** và chờ kết quả. Web hiển thị:

- ảnh gốc;
- vùng thùy phổi màu xanh;
- vùng tổn thương màu đỏ;
- tỷ lệ diện tích tổn thương trên tổng diện tích phổi.

Lần dự đoán đầu tiên chậm hơn vì phải nạp nhiều model vào GPU. Ảnh giống hệt đã
chạy trước đó được đọc từ cache trong `artifacts/web_v75_cache`.

Dừng web bằng `Ctrl+C` trong terminal.

### Cho máy khác trong cùng mạng truy cập

Chỉ dùng khi mạng nội bộ đáng tin cậy:

```bash
PYTHONPATH=src python -m uvicorn web_app:app \
  --host 0.0.0.0 --port 8785
```

Sau đó mở `http://IP_CUA_MAY_CHAY_MODEL:8785` từ máy khác. Firewall phải cho
phép TCP port 8785. Web hiện không có đăng nhập, vì vậy không mở trực tiếp ra
Internet.

## 6. Định dạng input và output

- Input phải là ảnh mà Pillow đọc được, ví dụ PNG hoặc JPEG.
- Kích thước mỗi chiều phải từ 64 pixel trở lên.
- Ảnh được chuyển sang RGB và letterbox về 512×512 trong lúc inference.
- Mask lesion luôn bị giới hạn bên trong mask phổi.
- Tỷ lệ được tính trên kích thước ảnh output đã phục hồi.
- Web không sửa ảnh nguồn.

## 7. Chạy test

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -q -p no:cacheprovider
```

Máy nguồn gần nhất đạt **53/53 test**. Nếu máy mới chưa có pytest:

```bash
python -m pip install pytest
```

Test không thay thế cho một lần dự đoán ảnh thật bằng GPU.

## 8. Lỗi thường gặp

### `GPU available: False`

Kiểm tra `nvidia-smi`, NVIDIA driver, cấu hình GPU của WSL2 và bản PyTorch CUDA.
Không cài nhầm bản PyTorch CPU-only.

### `CUDA out of memory`

Tắt các chương trình đang chiếm GPU rồi khởi động lại web. Không chạy nhiều
request cùng lúc; web đã khóa xử lý tuần tự trong một process.

### `ModuleNotFoundError`

Đảm bảo `.venv` đã activate, đã chạy `pip install -r
requirements-inference.txt`, đang đứng đúng thư mục và có `PYTHONPATH=src`.

### Không tìm thấy checkpoint/model

Không chỉ sao chép `src`. Cần sao chép nguyên folder, đặc biệt là `artifacts`,
`models` và `vendor/sam2`. Không đổi tên folder gốc.

### Port 8785 đang được sử dụng

```bash
PYTHONPATH=src python -m uvicorn web_app:app \
  --host 127.0.0.1 --port 8786
```

Sau đó mở `http://127.0.0.1:8786`.

## 9. Kết quả validation hiện tại

V75 trên 8 ảnh regression-validation:

| Chỉ số | Phổi | Tổn thương |
| --- | ---: | ---: |
| Dice | 99,64% | 80,54% |
| Precision | 99,91% | 80,53% |
| Recall | 99,37% | 80,55% |
| Specificity | 99,91% | 95,62% |

Lesion TP/FP/FN/TN theo pixel: `104.225 / 25.194 / 25.173 / 550.537`.
Lesion-ratio MAE là 2,94 điểm phần trăm.

Cohort này đã tham gia chọn blend/ngưỡng, nên đây là validation/regression chứ
không phải kết quả test độc lập hay cam kết độ chính xác trên mọi ảnh mới.

## 10. Ghi chú kỹ thuật

V75 ensemble bốn nhánh V40/V64/V67/V74. Toàn bộ checkpoint production, SAM2,
DINOv2 và lung U-Net đã được đóng gói trong folder. Inference web không phụ
thuộc file project nào nằm ngoài `Unsupervised_Lesion_Workflow`; chỉ các lệnh
train/validation lịch sử mới cần dataset/reference bên ngoài.
