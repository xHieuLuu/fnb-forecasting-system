# Đặc tả thiết kế: Pipeline Tích hợp & Phân rã Dữ liệu Thực tế Cửa hàng Cà phê/Trà sữa (Bộ Dữ liệu Hiệu chuẩn 14 Tháng)

- **Ngày lập:** 2026-08-26
- **Trạng thái:** Đã được người dùng phê duyệt qua quy trình Brainstorming
- **Mã định danh:** `real_store_calibrated_14m`
- **Mục tiêu:** Xử lý dữ liệu thực tế từ file `dataset.xlsx`, vá lỗi dữ liệu, phân rã doanh thu tổng thành số ly chi tiết theo từng món và sinh chuỗi dữ liệu 14 tháng hiệu chuẩn chuẩn hóa tương thích hoàn toàn với hệ thống `fnb-forecasting`.

---

## 1. Bối cảnh và Vấn đề Kỹ thuật

### 1.1. Dữ liệu thực tế đầu vào (`dataset.xlsx`)
File dữ liệu thực tế được thu thập từ một quán cà phê/trà sữa quy mô nhỏ phục vụ giới trẻ/sinh viên tại TP. Hồ Chí Minh với các đặc điểm:
- **Sheet Doanh thu (Tháng 5, Tháng 6, Tháng 7):**
  - Tháng 5 & Tháng 6/2025 ghi nhận doanh thu thực tế dao động từ **50.000đ đến 585.000đ/ngày** (trung bình ~250k - 300k/ngày, tương đương 12 - 25 ly/ngày). Có ghi nhận chi phí mua đá bi hàng ngày (~10k - 30k/ngày) và tiền tip/lương nhân viên.
  - **Lỗi dữ liệu thực tế:** Sheet `Tháng 7` là bản sao chép (copy-paste) $100\%$ dữ liệu của `Tháng 5`. Thực chất chỉ có 2 tháng dữ liệu thực nghiệm độc lập (T5 và T6/2025).
  - Dữ liệu bán hàng chỉ có **tổng doanh thu/ngày** theo các hình thức (Tiền mặt, MoMo, Chuyển khoản), không có lịch sử chi tiết số ly bán ra của từng món ($q_{i,t}$).
- **Sheet Định mức (`giá costcf`):**
  - Chứa công thức pha chế của ~25 món và bảng giá nhập nguyên vật liệu.
  - **Lỗi công thức:** Xuất hiện lỗi `#DIV/0!` do thiếu số lượng đóng gói hoặc giá nhập của một số nguyên vật liệu (Bột cacao, Mứt vải, Sirup vải, bao bì ly túi, ống hút).
  - Giá bán trong file Excel cũ có chênh lệch so với Bảng giá hoạt động chuẩn cập nhật mới nhất.

### 1.2. Thách thức và Giải pháp
Để đưa dữ liệu thực tế này vào huấn luyện mô hình Machine Learning / Deep Learning (LightGBM, LSTM, TFT, ETS) và chạy hệ thống gợi ý đặt hàng (BOM & Inventory Optimization), cần giải quyết 3 bài toán:
1. **Vá lỗi Master Data & Bảng giá:** Chuẩn hóa toàn bộ 43 món theo bảng giá mới, sửa triệt để các lỗi `#DIV/0!`, điền đầy đủ giá nhập thị trường TP.HCM.
2. **Bài toán Phân rã Nghịch đảo (Disaggregation Problem):** Dùng thuật toán lấy mẫu xác suất nguyên có ràng buộc để phân rã tổng doanh thu $Y_t$ của Tháng 5 & Tháng 6 thành số ly $q_{i,t}$ khớp chính xác $100\%$ số tiền thực tế.
3. **Mở rộng Chuỗi Thời gian 14 Tháng (01/05/2025 – 30/06/2026 = 426 ngày):** Tái tạo chu kỳ thời gian đầy đủ (Seasonal Cycle) gắn liền với lịch sinh viên TP.HCM, biến động thời tiết nắng mưa Sài Gòn, các dịp lễ và Tết Nguyên Đán Bính Ngọ 2026.

---

## 2. Danh mục Thực đơn (Menu) & Bảng Định mức (BOM) Chuẩn hóa

### 2.1. Danh mục 43 Sản phẩm (`item_master`)

| Nhóm danh mục | Mã món (`item_id`) | Tên món | Giá bán (VNĐ) |
| :--- | :--- | :--- | :--- |
| **Coffee** | `CF_DEN` | Cà phê đen | 15.000 |
| | `CF_SUA` | Cà phê sữa | 17.000 |
| | `CF_BACXIU` | Bạc xỉu | 20.000 |
| | `CF_TRUNG` | Cà phê trứng | 25.000 |
| | `CF_MUOI` | Cà phê muối | 20.000 |
| | `CF_BUONME` | Cà phê Buôn Mê | 25.000 |
| | `CF_TIRAMISU` | Cà phê Tiramisu | 25.000 |
| **Topping** | `TOP_TCDEN` | Trân châu đen | 5.000 |
| | `TOP_TCTRANG` | Trân châu trắng | 5.000 |
| | `TOP_KEMUOI` | Kem muối | 7.000 |
| | `TOP_KEMTRUNG` | Kem trứng | 10.000 |
| | `TOP_KEMBUONME` | Kem Buôn Mê | 10.000 |
| **Olong / Soda / Tea** | `TEA_DAU` | Trà Olong Dâu | 20.000 |
| | `TEA_VIETQUAT` | Trà Olong Việt Quất | 20.000 |
| | `TEA_KIWI` | Trà Olong Kiwi | 20.000 |
| | `TEA_CHANHDAY` | Trà Olong Chanh Dây | 20.000 |
| | `TEA_LUUDO` | Trà Olong Lựu Đỏ | 20.000 |
| | `TEA_OIHONG` | Trà Olong Ổi Hồng | 20.000 |
| | `TEA_TAOXANH` | Trà Olong Táo Xanh | 20.000 |
| | `TEA_DAOCAMSA` | Trà Đào Cam Sả | 20.000 |
| | `TEA_DUALUOI` | Trà Olong Dưa Lưới | 20.000 |
| | `TEA_MANGCAU` | Trà Mãng Cầu | 20.000 |
| | `TEA_HONGTRA` | Hồng Trà Truyền Thống | 15.000 |
| **Iceblend / Đá xay** | `ICE_CF_CARAMEL` | Coffee Caramel Đá Xay | 30.000 |
| | `ICE_TRAICAY` | Đá Xay Dâu / Chocolate / Việt Quất | 30.000 |
| | `ICE_MATCHA` | Matcha Đá Xay | 30.000 |
| | `ICE_OREO` | Oreo Đá Xay | 30.000 |
| | `ICE_MILO` | Milo Dầm Trân Châu | 25.000 |
| **Milktea / Trà sữa** | `MT_TRUYENTHONG` | Trà Sữa Truyền Thống | 15.000 |
| | `MT_DUONGDEN` | Sữa Tươi Trân Châu Đường Đen | 20.000 |
| | `MT_THAI` | Trà Sữa Thái Xanh / Đỏ | 20.000 |
| | `MT_DAU_VQ` | Trà Sữa Dâu / Việt Quất | 20.000 |
| | `MT_MATCHA` | Trà Sữa Matcha | 20.000 |
| | `MT_KHOAIMON` | Trà Sữa Khoai Môn | 20.000 |
| | `MT_TAOXANH` | Trà Sữa Táo Xanh | 20.000 |
| | `MT_COMDEO` | Trà Sữa Cốm Dẻo | 25.000 |
| | `MT_OLONG_LAI` | Olong Lài Sữa | 20.000 |
| **Matcha / Cacao / Dessert**| `MAT_LATTE` | Matcha Latte | 25.000 |
| | `MAT_GAU` | Matcha Sữa Gấu | 30.000 |
| | `MAT_FRUIT` | Matcha Dâu / Việt Quất / Khoai Môn | 30.000 |
| | `CAC_LATTE` | Cacao Latte | 25.000 |
| | `CAC_GAU` | Cacao Sữa Gấu | 30.000 |
| | `DES_PANNACOTTA` | Panna Cotta | 15.000 |

---

### 2.2. Danh mục Nguyên vật liệu & Vá lỗi `#DIV/0!` (`ingredient_master`)

Các nguyên liệu bị thiếu đơn giá trong file Excel được vá theo giá chuẩn thị trường F&B TP.HCM:
- **Bột Cacao:** Giá nhập 85.000đ / gói 500g $\rightarrow$ 170đ / gam (`shelf_life`: 180 ngày).
- **Mứt Vải & Sirup Vải:** Mứt vải 95.000đ / hũ 1kg (95đ/g); Sirup vải 65.000đ / chai 700ml (92.86đ/ml).
- **Bao bì tiêu hao:**
  - `PACK_CUP500`: Ly + Nắp 500ml: 983đ / bộ.
  - `PACK_STRAW`: Ống hút lớn/nhỏ: 150đ / cái.
  - `PACK_BAG`: Túi chữ T / Túi đựng mang về: 100đ / cái.
  - `PACK_SPOON`: Muỗng nhựa: 120đ / cái.
- **Đá bi lạnh:** Tiêu thụ trung bình 15.000đ – 25.000đ/ngày theo hóa đơn thực tế.

---

## 3. Động lực Học Chuỗi Thời gian & Phân rã 14 Tháng (05/2025 – 06/2026)

### 3.1. Thuật toán Phân rã Doanh thu Thực tế (Tháng 5 & Tháng 6/2025)
- Với mỗi ngày $t \in \text{Tháng 5, Tháng 6/2025}$:
  Ta giải bài toán quy hoạch nguyên có trọng số ưu tiên:
  $$\text{minimize} \quad D_{KL}(p_t \parallel p_0) \quad \text{sao cho} \quad \sum_{i=1}^{43} P_i \cdot q_{i,t} = Y_t^{\text{excel}}, \quad q_{i,t} \in \mathbb{N}_{\ge 0}$$
  Trong đó $p_0$ là vector phân phối thị hiếu khách hàng sinh viên/giới trẻ (Trà sữa, Trà trái cây, Bạc xỉu, Topping chiếm tỷ trọng cao).

### 3.2. Mô hình hóa Chu kỳ 14 Tháng (426 ngày)
- **T5/2025 – T6/2025:** Khớp chính xác $100\%$ doanh thu và chi phí thực tế từ Excel.
- **T7/2025 – T8/2025 (Nghỉ hè sinh viên):** Doanh thu ngày thường giảm nhẹ ($\approx 15\%$), duy trì khách buổi tối.
- **T9/2025 – T10/2025 (Tựu trường & Khai giảng):** Doanh thu tăng trưởng mạnh mẽ $+25\% - 35\%$, nhu cầu trà sữa & trà trái cây bùng nổ.
- **T11/2025 – T12/2025 (Mùa thi cử & Lễ hội cuối năm):** Khách ngồi học nhóm lâu; tăng trưởng cà phê, bạc xỉu và đồ uống mùa lễ hội.
- **T1/2026 – T2/2026 (Tết Nguyên Đán Bính Ngọ 2026):**
  - Tuần cận Tết: Nhu cầu mua mang đi tăng vọt.
  - Nghỉ Tết (29 Tết đến mùng 3 Tết): Đóng cửa quán (`store_open = False`, doanh số = 0).
  - Khai xuân (mùng 4 trở đi): Mở hàng đầu năm, sinh viên quay lại thành phố.
- **T3/2026 – T6/2026 (Mùa khô nóng đỉnh điểm TP.HCM):** Nhiệt độ 35°C–38°C kích cầu các dòng Đá xay, Trà mãng cầu, Trà dâu, Milo dầm tăng $+40\%$.

### 3.3. Tích hợp Thời tiết & Tồn kho thực tế
- **Nhiệt độ & Lượng mưa:** Chuỗi thời tiết TP.HCM với mùa mưa (T5 - T11, mưa rào chiều làm giảm khách vãng lai) và mùa khô (T12 - T4).
- **Sự cố thiếu hàng thực tế (Stockout events):** Mô phỏng xác suất nhỏ hết nguyên liệu (như trân châu đen tối CN) để mô hình AI học tính năng bù đắp nhu cầu tiềm năng (censored demand).

---

## 4. Cấu trúc 8 Bảng Dữ liệu Chuẩn (`DataBundle`)

Dataset được tạo ra tại thư mục: `data/processed/real_store_14m/`

```
data/processed/real_store_14m/
├── daily_sales.csv         # 43 món x 426 ngày = 18.318 dòng
├── item_master.csv         # 43 sản phẩm với mã, tên, danh mục, ngày ra mắt
├── recipes.csv             # Bảng định lượng nguyên liệu (BOM) chi tiết
├── ingredient_master.csv   # Danh mục nguyên vật liệu, quy cách, HSD, lead time
├── calendar.csv            # 426 ngày với cờ ngày lễ VN, Tết, thứ trong tuần, store_open
├── weather.csv             # Chuỗi thời tiết nhiệt độ, mưa, độ ẩm theo ngày
├── inventory_lots.csv      # Lô tồn kho ban đầu có ngày nhập và hạn sử dụng
├── scheduled_receipts.csv  # Lịch giao hàng từ nhà cung cấp
└── provenance.json         # Metadata ghi vết nguồn gốc, checksum, cờ dữ liệu
```

---

## 5. Quy trình Kiểm thử & Xác minh Tính đúng đắn (Verification Plan)

1. **Kiểm tra Schema & Data Contract (`validate_bundle`):**
   - Đảm bảo 100% 8 bảng vượt qua bộ kiểm tra schema của thư viện `fnb_forecast.data.validation`.
   - Không chứa giá trị âm, không có giá trị null vào những ngày quán mở cửa (`store_open = True`).
2. **Kiểm tra Tính nhất quán Tài chính:**
   - Tổng doanh thu các ngày trong Tháng 5 và Tháng 6/2025 từ `daily_sales.csv` phải bằng chính xác tổng doanh thu trong file `dataset.xlsx`.
3. **Kiểm thử Pipeline Mô hình AI:**
   - Huấn luyện thử nghiệm LightGBM, ETS, LSTM trên dataset `real_store_14m`.
   - Chạy lệnh dự báo 7 ngày (`fnb-forecast forecast`) và mô phỏng đặt hàng tồn kho (`fnb-forecast simulate-inventory`).
4. **Kiểm tra Dashboard Streamlit:**
   - Khởi chạy `streamlit run src/fnb_forecast/app/Home.py` để trực quan hóa dữ liệu và kiểm tra trực tiếp giao diện.
