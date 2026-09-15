# XÂY DỰNG HỆ THỐNG DỰ BÁO NHU CẦU MÓN, DOANH THU VÀ NGUYÊN LIỆU CHO CỬA HÀNG CÀ PHÊ/TRÀ SỮA
*(F&B Forecasting System - Calibrated Real-Store 14-Month Dataset)*

---

## 📌 Giới Thiệu Đồ Án

Đồ án xây dựng nền tảng dự báo chuỗi thời gian (Time-series forecasting) chuyên biệt cho ngành thực phẩm & đồ uống (F&B), giúp dự báo nhu cầu tiêu thụ món, tổng doanh thu cửa hàng và tự động quy đổi nhu cầu nguyên liệu đầu vào. 

Hệ thống được tinh chỉnh (calibrated) dựa trên dữ liệu bán hàng 14 tháng thực tế của chuỗi cửa hàng Cà phê & Trà sữa tại TP.HCM (từ 01/05/2025 đến 30/06/2026), tích hợp cơ chế dự báo an toàn (fallback state), kiểm toán nguồn gốc dữ liệu (provenance) và đề xuất điểm đặt hàng tồn kho (Safety Stock / Reorder Point / FEFO).

---

## ✨ Các Tính Năng Chính

- **Dự Báo Đa Mục Tiêu (Multi-target Forecasting)**:
  - **Dự báo số lượng từng món (Item level)**: 43 mặt hàng (Cà phê, Trà sữa, Bánh pastry, Trà trái cây...).
  - **Dự báo doanh thu (Revenue level)**: Dự báo tổng doanh thu toàn cửa hàng theo từng ngày.
  - **Dự báo nguyên liệu (Ingredient level)**: Tự động bóc tách định mức (BOM - Bill of Materials) từ nhu cầu món ra lượng nguyên liệu cần dùng (sữa tươi, hạt cà phê, bắp, đường, siro...).
- **Xử Lý Dữ Liệu Thực Tế (ETL Pipeline)**:
  - Tự động chuẩn hóa file POS Excel xuất ra 8 bảng dữ liệu chuẩn (`DataBundle`).
  - Tích hợp các yếu tố ngoại cảnh: Thời tiết TP.HCM (nhiệt độ, lượng mưa), lịch học sinh/sinh viên, ngày lễ Tết Nguyên Đán, chương trình khuyến mãi và cờ đánh dấu đứt hàng (stockout).
- **Huấn Luyện & Kiểm Đánh Giá Mô Hình**:
  - Hỗ trợ đa dạng thuật toán: Exponential Smoothing, ARIMA/SARIMAX, LightGBM, Prophet, Ensemble.
  - Tự động chọn mô hình tối ưu theo Validation Leaderboard và Test Leaderboard.
  - Cơ chế Fallback an toàn (Naive/Moving Average) khi xảy ra lỗi dữ liệu hoặc thiếu lịch sử.
- **Quản Lý Tồn Kho & Gợi Ý Đặt Hàng**:
  - Tính toán Tồn kho an toàn (Safety Stock) theo Service Level (ví dụ 95%).
  - Gợi ý lịch đặt hàng nguyên liệu và cảnh báo hạn sử dụng lô hàng (FEFO/FIFO).
- **Giao Diện Trực Quan (Streamlit Dashboard)**:
  - Trực quan hóa kết quả dự báo 7 ngày tiếp theo.
  - Hiển thị nguồn gốc dữ liệu (provenance), cờ cảnh báo rủi ro và theo dõi biến động doanh thu/nguyên liệu mà không làm thay đổi trạng thái huấn luyện từ UI.

---

## 🛠️ Cài Đặt & Cấu Hình Môi Trường

### Yêu cầu hệ thống:
- **Python**: 3.10 trở lên
- **Hệ điều hành**: Windows / macOS / Linux

### Các bước cài đặt:

```powershell
# 1. Tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Trên Windows PowerShell
# source .venv/bin/activate    # Trên Linux/macOS

# 2. Cài đặt dự án ở chế độ Editable kèm thư viện phụ thuộc
python -m pip install -e ".[dev]"
```

Cấu hình mặc định của hệ thống nằm tại file `configs/base.yaml`.

---

## 📊 Quy Trình Chạy Pipeline & Thử Nghiệm

Chạy các câu lệnh bên dưới từ thư mục gốc của dự án theo đúng thứ tự:

### 1. Chuyển đổi dữ liệu POS Excel sang DataBundle (8 bảng)
Chuyển đổi dữ liệu 14 tháng thực tế (`dataset_real_store_14m_monthly.xlsx` hoặc `dataset.xlsx`):
```powershell
fnb-forecast generate-real-store --excel dataset.xlsx --output data/processed/real_store_14m
```
*Lệnh này sẽ tự động phân rã doanh thu lịch sử, sửa chữa dữ liệu công thức món (BOM), tích hợp thời tiết TP.HCM, lịch nghỉ lễ Tết và tạo 8 bảng CSV chuẩn hóa tại `data/processed/real_store_14m/`.*

### 2. Huấn luyện mô hình (Model Training)
```powershell
fnb-forecast train --dataset real_store_14m --config configs/base.yaml --profile full
```

### 3. Đánh giá bảng xếp hạng mô hình (Evaluation & Leaderboard)
```powershell
# Đánh giá tập Validation
fnb-forecast evaluate --dataset real_store_14m --role validation

# Đánh giá tập Test (chỉ dành cho mô hình được chọn)
fnb-forecast evaluate --dataset real_store_14m --role test --selected-only
```

### 4. Tạo dự báo 7 ngày & Đề xuất nguyên liệu
```powershell
fnb-forecast forecast --dataset real_store_14m --origin 2026-05-01 --service-level 0.95 --output artifacts/forecasts/2026-05-01
```
*Kết quả sẽ xuất ra các file: `item_forecasts.csv`, `revenue_forecast.csv`, `ingredient_forecast.csv`, `order_proposals.csv` và `run_metadata.json` trong thư mục `artifacts/forecasts/2026-05-01/`.*

### 5. Mô phỏng vận hành tồn kho (Inventory Simulation)
```powershell
fnb-forecast simulate-inventory --dataset real_store_14m --service-level 0.95 --output artifacts/reports/inventory
```

---

## 🖥️ Hướng Dẫn Chạy Giao Diện Dashboard (Streamlit)

Chạy ứng dụng Web Dashboard bằng 1 trong các cách sau:

**Cách 1 (Từ thư mục làm việc PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1; streamlit run src/fnb_forecast/app/Home.py
```

**Cách 2 (Chạy trực tiếp file thực thi Streamlit):**
```powershell
.\.venv\Scripts\streamlit.exe run src/fnb_forecast/app/Home.py
```

Dashboard tự động đọc các file dự báo mới nhất tại `artifacts/forecasts/`, hiển thị chi tiết provenance, cảnh báo rủi ro và minh họa đồ thị trực quan.

---

## 📁 Cấu Trúc Dữ Liệu Đầu Vào (DataBundle)

Bộ dữ liệu đầu vào chuẩn hóa bao gồm 8 bảng CSV (mã hóa UTF-8) và 1 file metadata `provenance.json`:
1. `daily_sales.csv`: Nhật ký bán hàng hàng ngày (mã món, ngày, số lượng, giá bán, cờ khuyến mãi, cờ đứt hàng, cờ mở cửa).
2. `item_master.csv`: Danh mục món ăn/đồ uống và phân loại nhóm hàng.
3. `calendar.csv`: Lịch ngày, cờ cuối tuần, ngày lễ Tết Nguyên Đán, cờ lịch học sinh/sinh viên.
4. `weather.csv`: Dữ liệu thời tiết TP.HCM (nhiệt độ trung bình, lượng mưa).
5. `recipes.csv`: Định mức nguyên liệu (BOM) cho từng món.
6. `ingredient_master.csv`: Danh mục nguyên liệu, đơn vị tính, đơn giá, thời gian giao hàng (Lead time), MOQ.
7. `inventory_lots.csv`: Quản lý lô nguyên liệu tồn kho và hạn sử dụng (FEFO).
8. `scheduled_receipts.csv`: Các đơn hàng nguyên liệu đã đặt đang trên đường về.

---

## 🧪 Kiểm Thử (Testing)

Chạy bộ kiểm thử tự động để đảm bảo tính đúng đắn của hệ thống:
```powershell
pytest
```

---

## ⚠️ Lưu Ý & Giới Hạn

- **Độ tin cậy & Khoảng dự báo**: Khoảng tin cậy dự báo được tính toán dựa trên sai số validation (Residual-based bounds).
- **Quyền riêng tư & Bảo mật**: Dữ liệu huấn luyện đã được loại bỏ thông tin nhạy cảm của cửa hàng.
- **Git Hygiene**: Các thư mục dữ liệu sinh ra trong quá trình chạy (`data/processed/`, `artifacts/`, `.cache/`, `.venv/`) được cấu hình tự động loại trừ khỏi Git theo file `.gitignore`.
