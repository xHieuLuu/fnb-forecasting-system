# Đặc tả thiết kế: Dự báo nhu cầu nguyên vật liệu và doanh thu cho cửa hàng F&B

Ngày: 2026-08-13

Trạng thái: Đã được người dùng duyệt; hiệu chỉnh nguồn weather backtest theo tài liệu API chính thức

Phạm vi: Đồ án cá nhân ngành Khoa học Máy tính, thời gian 12–16 tuần

## 1. Tên đề tài

**Nghiên cứu và xây dựng hệ thống dự báo đa bước nhu cầu món, nguyên vật liệu và doanh thu cho cửa hàng cà phê/trà sữa bằng mô hình Deep Learning chuỗi thời gian**

Tên tiếng Anh đề xuất:

**Research and Development of a Multi-Horizon Deep Learning System for Menu Demand, Ingredient Requirements, and Revenue Forecasting in a Coffee and Milk Tea Shop**

## 2. Bối cảnh và vấn đề

Cửa hàng cà phê/trà sữa quy mô nhỏ thường phải cân bằng hai rủi ro: nhập thiếu gây mất doanh thu và nhập dư gây tồn kho hoặc hủy nguyên liệu hết hạn. Nhu cầu biến động theo thứ trong tuần, ngày lễ, thời tiết, giá bán và khuyến mãi. Dự báo tổng doanh thu đơn thuần không đủ để hỗ trợ nhập hàng, vì quản lý cần biết số lượng từng món và lượng từng nguyên liệu.

Đồ án xây dựng một prototype nghiên cứu cho một cửa hàng giả định tại TP.HCM. Hệ thống dự báo số lượng từng món theo ngày trong 7 ngày tiếp theo. Dự báo món được quy đổi nhất quán sang doanh thu và nguyên liệu bằng giá bán đã lên kế hoạch cùng bảng công thức định lượng (bill of materials, BOM). Một bộ lập kế hoạch tồn kho sau đó đề xuất lượng nhập có xét tồn an toàn, tồn khả dụng, hàng đang về, lead time, quy cách mua và hạn sử dụng.

Do không có dữ liệu từ cửa hàng thật tại Việt Nam, đồ án tách rõ:

- **Bằng chứng định lượng:** thí nghiệm tái lập trên benchmark bán hàng thực, công khai.
- **Minh họa ứng dụng Việt Nam:** kịch bản tổng hợp có gắn nhãn rõ, dùng lịch Việt Nam và dữ liệu thời tiết TP.HCM.

Đồ án không tuyên bố đã chứng minh hiệu quả thương mại tại một cửa hàng Việt Nam thật.

## 3. Mục tiêu

### 3.1. Mục tiêu chính

1. So sánh phương pháp baseline, thống kê, Machine Learning và Deep Learning trong dự báo nhu cầu từng mặt hàng cho 7 ngày tiếp theo.
2. Đánh giá đóng góp của lịch, ngày lễ, giá, khuyến mãi và thời tiết bằng thí nghiệm ablation.
3. Quy đổi dự báo món thành dự báo nguyên liệu và doanh thu theo một mô hình dữ liệu nhất quán.
4. Đánh giá tác động của dự báo lên quyết định nhập hàng bằng mô phỏng tồn kho.
5. Cung cấp dashboard Streamlit để người quản lý xem dự báo, cập nhật tồn kho và xuất đề xuất nhập hàng.

### 3.2. Mục tiêu kỳ vọng, không phải giả định kết quả

Mô hình được chọn được kỳ vọng giảm WAPE ít nhất 5% so với Seasonal Naive trên benchmark. Nếu Deep Learning không vượt ETS hoặc LightGBM, kết luận đó vẫn hợp lệ và phải được báo cáo trung thực.

## 4. Câu hỏi và giả thuyết nghiên cứu

### 4.1. Câu hỏi nghiên cứu

- **RQ1:** Global LSTM và Temporal Fusion Transformer có cải thiện dự báo nhu cầu 7 ngày so với Seasonal Naive, ETS và LightGBM không?
- **RQ2:** Trong kịch bản tổng hợp Việt Nam, lịch, ngày lễ, giá, khuyến mãi và thời tiết cải thiện độ chính xác ở mức nào?
- **RQ3:** Trong mô phỏng tồn kho của kịch bản tổng hợp, chính sách nhập hàng dựa trên dự báo có cải thiện fill rate, số ngày thiếu hàng và tỷ lệ hủy hàng so với chính sách dùng trung bình bán 7 ngày gần nhất không?

### 4.2. Giả thuyết kiểm chứng

- **H1:** Mô hình global học chung nhiều mặt hàng đạt WAPE thấp hơn Seasonal Naive ở cấp mặt hàng–ngày.
- **H2:** Nhóm biến ngoại sinh làm giảm sai số rõ nhất quanh cuối tuần, ngày lễ, khuyến mãi và các ngày thời tiết cực đoan.
- **H3:** Chính sách nhập dựa trên dự báo giảm stockout mà không làm waste rate tăng quá mức so với baseline vận hành.

## 5. Phạm vi

### 5.1. Trong phạm vi

- Một cửa hàng cà phê/trà sữa giả định tại TP.HCM.
- 15 món đại diện thuộc các nhóm cà phê, trà trái cây, trà sữa và đồ uống khác.
- 15 nguyên liệu và vật tư quan trọng, gồm cả nguyên liệu dễ hỏng và bao bì.
- Dữ liệu theo ngày và dự báo trực tiếp 7 ngày tiếp theo.
- Năm mô hình: Seasonal Naive, ETS, LightGBM, Global LSTM và Temporal Fusion Transformer (TFT).
- Backtesting cuốn chiếu, ablation feature và đánh giá trên tập test cuối chưa sử dụng khi chọn mô hình.
- BOM, dự báo doanh thu, mô phỏng tồn kho và đề xuất nhập hàng.
- Dashboard Streamlit chạy cục bộ và hỗ trợ CSV đúng schema.

### 5.2. Ngoài phạm vi

- Chuỗi nhiều cửa hàng, điều chuyển tồn kho hoặc tối ưu mạng lưới phân phối.
- Lập lịch nhân sự, lựa chọn nhà cung cấp hoặc tối ưu tuyến giao hàng.
- Tối ưu chi phí đa mục tiêu ở cấp nghiên cứu vận trù học.
- Đăng nhập, phân quyền, thanh toán, triển khai production hoặc tích hợp trực tiếp POS.
- Khẳng định quan hệ nhân quả giữa thời tiết, khuyến mãi và nhu cầu.
- Khẳng định hiệu quả thực tế tại thị trường Việt Nam khi chưa có pilot với cửa hàng thật.

## 6. Các hướng đã cân nhắc

### 6.1. Dự báo từng món rồi quy đổi sang nguyên liệu và doanh thu — được chọn

Ưu điểm là nhu cầu món, nguyên liệu và doanh thu luôn nhất quán; quản lý có thể truy ngược một đề xuất nhập hàng về các món tạo ra nhu cầu. Nhược điểm là sai số ở cấp món truyền sang nguyên liệu, nhưng có thể đo trực tiếp trong backtesting và mô phỏng.

### 6.2. Dự báo trực tiếp từng nguyên liệu và doanh thu — không chọn

Pipeline ngắn hơn nhưng kết quả nguyên liệu có thể mâu thuẫn với số món bán, khó giải thích và khó tái sử dụng khi công thức thay đổi.

### 6.3. Dự báo tổng nhu cầu rồi phân bổ xuống món — không chọn

Chuỗi tổng ổn định hơn nhưng cần một mô hình phân bổ thứ hai, làm tăng rủi ro sai số và độ phức tạp không cần thiết cho đồ án cá nhân 12–16 tuần.

## 7. Chiến lược dữ liệu

### 7.1. Lớp A: benchmark công khai

Đồ án dùng dữ liệu [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), chứa 1.969 ngày bán hàng ở cấp sản phẩm–cửa hàng, cùng giá và lịch sự kiện. Từ một cửa hàng, hệ thống chọn cố định 20 SKU thuộc nhóm `FOODS` có lịch sử đủ dài và tần suất bán cao. Việc chọn SKU chỉ dựa trên phần đầu của chuỗi dùng để huấn luyện, không nhìn tập validation hoặc test.

Vai trò của lớp A:

- So sánh năm mô hình trên chuỗi bán hàng thực.
- Kiểm tra backtesting, metric và khả năng tái lập.
- Đánh giá liệu mô hình phức tạp có thực sự tốt hơn baseline.

M5 là dữ liệu bán lẻ tại Hoa Kỳ, không phải dữ liệu quán cà phê Việt Nam. Kết quả trên M5 chỉ hỗ trợ kết luận về năng lực dự báo item-level, không chứng minh mức chính xác cho F&B Việt Nam.

### 7.2. Lớp B: kịch bản ứng dụng Việt Nam

Bộ sinh dữ liệu tạo 730 ngày liên tiếp từ 2024-01-02 đến 2025-12-31 cho 15 món của một quán giả định tại TP.HCM. Dữ liệu được lưu kèm trường `data_provenance=synthetic_vietnam_scenario` và được gắn nhãn tương tự trên dashboard, biểu đồ và bảng kết quả.

Nhu cầu được sinh bằng phân phối đếm quá phân tán, kết hợp:

- Mức nhu cầu cơ sở và độ phổ biến riêng của từng món.
- Quy luật theo thứ, cuối tuần, mùa và xu hướng dài hạn.
- Khoảng thời gian quanh Tết và ngày lễ Việt Nam.
- Độ nhạy theo nhiệt độ, mưa và độ ẩm khác nhau giữa các nhóm món.
- Giá bán, chương trình khuyến mãi đã lên lịch và thời điểm ra mắt món.
- Stockout, cú sốc ngẫu nhiên và thay đổi nhẹ về thị hiếu để tránh chuỗi quá lý tưởng.

Thời tiết quan sát từ [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) được dùng để sinh nhu cầu. Khi backtest, feature thời tiết tương lai phải lấy từ [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api): biến hậu tố `_previous_day1` đến `_previous_day7` biểu diễn giá trị từng ngày đích được dự báo trước đúng 1–7 ngày. Dashboard vận hành dùng [Open-Meteo Forecast API](https://open-meteo.com/en/docs). Nếu previous-run tương ứng không khả dụng tại một cutoff, fold đó dùng biến thể mô hình không có thời tiết; không thay thế âm thầm bằng thời tiết quan sát thực tế.

Ngày lễ được tạo bằng thư viện [python-holidays](https://github.com/vacanza/holidays) với mã quốc gia `VN`, cộng thêm các feature khoảng cách đến/trôi qua Tết được tính chỉ từ lịch đã biết.

### 7.3. Menu đại diện

Kịch bản gồm 15 món: cà phê đen đá, cà phê sữa đá, bạc xỉu, espresso, latte, cappuccino, trà đào cam sả, trà vải, trà chanh, trà sữa truyền thống, trà sữa trân châu đường đen, trà sữa matcha, chocolate đá xay, matcha latte và nước cam.

Tên món và công thức nhằm tạo kịch bản vận hành quen thuộc tại Việt Nam, không được trình bày như thống kê thị trường hay công thức chuẩn của một thương hiệu.

Mười lăm nguyên liệu/vật tư được theo dõi là hạt cà phê, sữa đặc, sữa tươi, trà đen, trà xanh, bột matcha, syrup đường, đào ngâm, vải ngâm, cam tươi, chanh tươi, trân châu, bột chocolate, kem sữa và bộ ly–nắp–ống hút. Nước và đá không nằm trong bài toán tồn kho.

## 8. Mô hình dữ liệu

Các bảng lõi dùng đơn vị chuẩn và khóa rõ ràng để có thể thay dữ liệu tổng hợp bằng POS thật trong tương lai.

| Bảng | Trường chính | Mục đích |
|---|---|---|
| `daily_sales` | `date`, `item_id`, `quantity`, `unit_price`, `promo_flag`, `stockout_flag`, `store_open` | Doanh số theo món–ngày |
| `item_master` | `item_id`, `item_name`, `category`, `launch_date`, `active` | Danh mục món |
| `calendar` | `date`, `weekday`, `is_weekend`, `holiday_name`, `days_to_tet`, `days_after_tet` | Biến lịch biết trước |
| `weather` | `issued_at`, `target_date`, `temperature`, `rain`, `humidity`, `source` | Dự báo thời tiết có thời điểm phát hành |
| `recipes` | `item_id`, `ingredient_id`, `amount`, `unit`, `waste_rate` | BOM và hao hụt |
| `ingredient_master` | `ingredient_id`, `name`, `base_unit`, `pack_size`, `lead_time_days`, `review_period_days`, `shelf_life_days` | Quy tắc nhập hàng |
| `inventory_lots` | `ingredient_id`, `lot_id`, `on_hand`, `expiry_date`, `received_date` | Tồn kho theo lô |
| `scheduled_receipts` | `ingredient_id`, `arrival_date`, `quantity` | Hàng đã đặt đang về |
| `forecasts` | `forecast_origin`, `target_date`, `item_id`, `yhat`, `lower`, `upper`, `model_id` | Kết quả dự báo |

Các đại lượng nguyên liệu dùng đơn vị cơ sở như gram, millilitre hoặc cái. Việc đổi từ túi, chai, thùng sang đơn vị cơ sở diễn ra trước khi lập kế hoạch. Giá tiền dùng VND và lưu dưới dạng số nguyên.

## 9. Kiến trúc hệ thống

Hệ thống chia thành các thành phần nhỏ, độc lập:

1. **Data ingestion** đọc M5, dữ liệu tổng hợp hoặc CSV người dùng.
2. **Data validation** kiểm tra schema, khóa, kiểu dữ liệu, phạm vi giá trị, ngày thiếu và provenance.
3. **Vietnam scenario generator** tạo menu, chuỗi nhu cầu, giá, khuyến mãi, BOM và tồn kho có seed cố định.
4. **Feature pipeline** tạo feature chỉ từ dữ liệu khả dụng tại `forecast_origin`.
5. **Model adapters** cung cấp cùng giao diện `fit`, `predict` và `save/load` cho cả năm mô hình.
6. **Backtest engine** tạo các cutoff theo thời gian và lưu dự báo out-of-sample.
7. **Evaluator/model selector** tính metric, ablation, độ ổn định và chọn mô hình bằng validation WAPE.
8. **Forecast service** tải model đã chọn và sinh dự báo món cùng khoảng bất định.
9. **Revenue and inventory planner** quy đổi dự báo sang doanh thu, nguyên liệu và đề xuất nhập.
10. **Streamlit app** hiển thị dữ liệu, cảnh báo, kết quả nghiên cứu và cho phép xuất CSV.

Luồng chính:

```text
Nguồn dữ liệu
  → Kiểm tra dữ liệu
  → Feature pipeline
  → Backtesting / huấn luyện
  → Model registry
  → Dự báo 7 ngày theo món
  → BOM + giá + tồn kho
  → Doanh thu + nhu cầu nguyên liệu + đề xuất nhập
  → Dashboard / CSV
```

Huấn luyện và chọn mô hình là quy trình offline. Dashboard chỉ tải artifact đã được chọn; nút “Chạy dự báo mới” thực hiện inference và lập kế hoạch, không huấn luyện lại mô hình trong giao diện.

## 10. Thiết kế feature

### 10.1. Feature lịch sử

- Nhu cầu các lag 1, 7, 14, 21 và 28 ngày.
- Trung bình, độ lệch chuẩn, min và max trượt 7, 14 và 28 ngày; cửa sổ phải dịch một ngày trước khi tính.
- Giá, khuyến mãi và stockout trong quá khứ.

### 10.2. Feature biết trước

- Thứ, cuối tuần, tuần trong năm, tháng.
- Ngày lễ Việt Nam, số ngày đến Tết và số ngày sau Tết.
- Giá và khuyến mãi đã được lên lịch cho 7 ngày dự báo.
- Dự báo nhiệt độ, mưa và độ ẩm đúng phiên phát hành.

### 10.3. Feature tĩnh

- `item_id`, nhóm món và tuổi của món tính đến ngày dự báo.

Mọi transformer thống kê, scaler và encoder chỉ được fit trên phần train của từng fold.

## 11. Mô hình dự báo

### 11.1. Seasonal Naive

Dự báo `y(t+h)=y(t+h-7)` cho từng horizon. Đây là baseline bắt buộc và cũng là fallback cuối cùng khi artifact mô hình không dùng được.

### 11.2. ETS

Một mô hình ETS theo SKU với mùa vụ tuần. ETS đại diện cho phương pháp chuỗi thời gian thống kê và không dùng feature ngoại sinh.

### 11.3. LightGBM

Một mô hình global dạng bảng, dùng lag, rolling feature, feature tĩnh và feature biết trước. Dữ liệu huấn luyện được chuyển sang dạng item–forecast-origin–horizon; một LightGBM duy nhất học trực tiếp mục tiêu tương lai với `horizon` từ 1 đến 7 là một feature phân loại/thứ tự. Cách này cố định kiến trúc và tránh duy trì bảy model riêng.

### 11.4. Global LSTM

Embedding món và nhóm món được ghép với feature động. Encoder LSTM đọc cửa sổ lịch sử 28 hoặc 56 ngày; độ dài cửa sổ là một siêu tham số trong 12 cấu hình khai báo trước. Một head nhiều đầu ra dự báo trực tiếp 7 ngày bằng Huber loss. Đầu ra bị ràng buộc không âm.

### 11.5. Temporal Fusion Transformer

TFT dùng static embeddings, biến quá khứ và biến biết trước trong tương lai. Mô hình được triển khai bằng PyTorch/PyTorch Forecasting với cùng input window, horizon và cutoff như LSTM. Attention hoặc variable-selection weights chỉ được dùng để hỗ trợ diễn giải, không được trình bày như bằng chứng nhân quả.

### 11.6. Ngân sách chọn siêu tham số

Mỗi mô hình Deep Learning được thử tối đa 12 cấu hình khai báo trước trên validation folds; LightGBM tối đa 12 cấu hình. Mỗi cấu hình DL chạy ba seed cố định. Model selector chọn validation WAPE thấp nhất; nếu chênh lệch dưới 1%, ưu tiên mô hình ít tham số và thời gian inference thấp hơn.

## 12. Backtesting và đánh giá

### 12.1. Chia dữ liệu

- Giữ 28 ngày cuối làm tập test cuối cùng và không dùng để chọn feature, mô hình hoặc siêu tham số.
- Phần trước test dùng expanding-window validation với 8 cutoff cách nhau 7 ngày.
- Mỗi cutoff dự báo đúng 7 ngày.
- SKU, scaler, encoder và mọi thống kê được xác định từ train của từng fold.
- Kết quả cuối báo cáo trên 28 ngày test dưới dạng bốn forecast origins liên tiếp.

Mỗi lớp dữ liệu có leaderboard và model selector riêng. Kết quả M5 dùng để trả lời câu hỏi benchmark; model phục vụ dashboard được chọn chỉ bằng validation WAPE của kịch bản Việt Nam. Không chọn model dashboard dựa trên tập test hoặc giả định model thắng M5 cũng thắng dữ liệu kịch bản.

### 12.2. Metric

- **WAPE:** metric chọn mô hình chính, tính ở cấp item–day và cấp tổng cửa hàng.
- **MAE:** sai số tuyệt đối theo đơn vị bán.
- **RMSSE:** sai số chuẩn hóa theo seasonal-naive scale từ train.
- **Forecast Bias:** `sum(yhat-y)/sum(y)` để phát hiện dự báo dư hoặc thiếu.
- **Revenue WAPE:** tính sau khi nhân dự báo số lượng với giá bán đã biết.
- **Ingredient WAPE:** tính sau khi quy đổi cả actual và forecast qua cùng BOM.

MAPE không được dùng làm metric chính vì ngày có nhu cầu bằng 0 làm kết quả không ổn định. Chênh lệch WAPE giữa mô hình được báo cáo kèm khoảng tin cậy 95% bằng bootstrap trên các cặp SKU–forecast-window.

### 12.3. Ablation

Trên lớp B, ba cấu hình feature được so sánh trên cùng cutoff:

1. Chỉ lịch sử nhu cầu.
2. Lịch sử + lịch/ngày lễ Việt Nam.
3. Lịch sử + lịch/ngày lễ + giá/khuyến mãi + dự báo thời tiết.

Kết quả phân nhóm theo ngày thường/cuối tuần, quanh ngày lễ, có/không khuyến mãi và mưa/không mưa để trả lời RQ2.

Trên M5, ablation tương ứng gồm: chỉ lịch sử; lịch sử + calendar/event; và lịch sử + calendar/event + price. Kết quả của hai lớp dữ liệu được báo cáo tách biệt, không gộp thành một con số.

### 12.4. Khoảng dự báo

Dashboard hiển thị khoảng dự báo thực nghiệm 80%. Khoảng này được hiệu chỉnh từ residual out-of-sample trên validation theo nhóm món và horizon; không lấy residual từ train in-sample.

## 13. Quy đổi sang doanh thu và nguyên liệu

Với món `i`, nguyên liệu `k`, ngày `d`:

```text
forecast_revenue[d]
  = Σ_i forecast_item[i,d] × planned_unit_price[i,d]

ingredient_need[k,d]
  = Σ_i forecast_item[i,d]
      × recipe_amount[i,k]
      × (1 + waste_rate[k])
```

Giá khuyến mãi được phản ánh trong `planned_unit_price`. Hệ thống không huấn luyện một mô hình doanh thu riêng; điều này bảo đảm doanh thu luôn nhất quán với số món dự báo.

## 14. Đề xuất nhập hàng

Mỗi nguyên liệu có `lead_time_days` và `review_period_days`. Cửa sổ bảo vệ tồn kho là tổng hai giá trị này.

```text
coverage_days[k] = lead_time_days[k] + review_period_days[k]

target_stock[k]
  = forecast_need trong coverage_days[k]
  + safety_stock[k]

raw_order[k]
  = max(0,
      target_stock[k]
      - usable_on_hand[k]
      - scheduled_receipts[k])

suggested_order[k]
  = làm tròn raw_order[k] lên bội số pack_size[k]
```

Tồn an toàn được suy ra từ validation: quy đổi actual và forecast sang nguyên liệu, tính sai số thiếu `max(actual_need-forecast_need, 0)`, rồi cộng theo cửa sổ bảo vệ của nguyên liệu. Dashboard cho phép chọn phân vị 90%, 95% hoặc 98%; mặc định dùng phân vị 95%. Lần chạy phải lưu lựa chọn này trong metadata.

Nguyên liệu theo lô được xuất theo FEFO. `usable_on_hand` không tính phần tồn dự kiến hết hạn trước khi có thể sử dụng. Với nguyên liệu dễ hỏng, lượng nhập sau làm tròn bị giới hạn bởi nhu cầu dự báo trong hạn sử dụng cộng tồn an toàn; nếu quy cách đóng gói khiến giới hạn không thể thỏa, hệ thống vẫn đề xuất một pack tối thiểu và phát cảnh báo nguy cơ hủy.

### 14.1. Đánh giá vận hành

Mô phỏng tái hiện từng ngày trên tập test, dùng nhu cầu actual đã giữ lại. Hai chính sách dùng cùng tồn ban đầu, lịch nhận hàng, lead time, shelf life và pack size:

- **Forecast policy:** công thức trên với dự báo của model được chọn.
- **Baseline policy:** thay forecast bằng trung bình nhu cầu 7 ngày gần nhất.

Metric gồm fill rate, số ngày stockout, waste rate, average inventory và tổng lượng đặt. Kết luận không chỉ dựa vào một metric; giảm stockout bằng cách tăng tồn kho quá mức phải được chỉ rõ.

## 15. Dashboard Streamlit

Dashboard gồm bốn màn hình:

1. **Tổng quan:** doanh thu, tổng số ly, khoảng dự báo, cảnh báo kho và các việc cần xử lý trong 7 ngày.
2. **Dự báo món:** biểu đồ actual/forecast, khoảng bất định, bảng từng món và breakdown feature mang tính mô tả.
3. **Nhập nguyên liệu:** nhu cầu theo ngày, tồn hiện tại, lô sắp hết hạn, safety stock, lượng đề xuất và nút xuất CSV.
4. **Đánh giá mô hình:** leaderboard, backtesting folds, ablation, bias, provenance và giới hạn dữ liệu.

Luồng sử dụng:

```text
Chọn dữ liệu demo hoặc tải CSV đúng mẫu
  → kiểm tra lỗi và provenance
  → cập nhật tồn kho
  → chạy dự báo
  → xem cảnh báo
  → chọn service level 90%, 95% hoặc 98% (mặc định 95%)
  → xuất danh sách nhập hàng CSV
```

Ứng dụng chạy cục bộ. Không có đăng nhập, phân quyền hoặc tích hợp POS trực tiếp.

## 16. Xử lý lỗi và fallback

| Tình huống | Hành vi |
|---|---|
| Thiếu ngày trong `daily_sales` | Yêu cầu `store_open`; chỉ điền 0 khi cửa hàng mở, nếu không giữ trạng thái đóng cửa |
| Quantity, price hoặc inventory âm | Từ chối dữ liệu và báo bảng/dòng/cột lỗi |
| Trùng khóa item–date | Từ chối hoặc aggregate chỉ khi cấu hình nhập chỉ rõ dữ liệu ở cấp giao dịch |
| Không tải được dự báo thời tiết | Dùng cache; nếu cache không đủ thì dùng model variant không có weather và hiển thị cảnh báo |
| Món mới hoặc lịch sử dưới 28 ngày | Fallback về dự báo nhóm món theo tỷ trọng gần nhất; nếu không đủ dữ liệu nhóm thì Seasonal Naive hoặc median nhóm |
| Artifact model thiếu hoặc không tương thích schema | Không inference bằng artifact cũ; fallback Seasonal Naive và ghi log cảnh báo |
| Dự báo âm/NaN | Clamp giá trị âm về 0; NaN làm model run thất bại và kích hoạt fallback |
| Thiếu hạn sử dụng cho nguyên liệu dễ hỏng | Không áp dụng shelf-life cap và hiển thị cảnh báo bắt buộc |
| BOM thiếu cho một món | Vẫn hiển thị dự báo món/doanh thu nhưng loại món khỏi tổng nguyên liệu và cảnh báo thiếu BOM |

Fallback phải hiển thị trên dashboard và được ghi vào metadata của lần chạy; hệ thống không được âm thầm thay đổi mô hình.

## 17. Kiểm thử và tái lập

### 17.1. Unit tests

- BOM, hao hụt và quy đổi đơn vị.
- Tồn an toàn, coverage window, pack rounding và FEFO.
- Revenue calculation với giá khuyến mãi.
- Feature lag/rolling không đọc dữ liệu hiện tại hoặc tương lai.
- Split train/validation/test và cutoff không chồng lấn sai quy tắc.
- Metric với chuỗi bằng 0 và chuỗi gián đoạn.

### 17.2. Integration tests

- CSV hợp lệ → validation → feature → forecast → revenue → nguyên liệu → đề xuất nhập.
- CSV lỗi → thông báo đúng vị trí và không tạo forecast artifact.
- Weather API lỗi → cache/model fallback hoạt động và metadata ghi đúng.
- Model artifact lỗi → Seasonal Naive fallback hoạt động.

### 17.3. Reproducibility checks

- Seed của Python, NumPy và PyTorch được cố định trong cấu hình.
- Phiên bản thư viện, checksum dữ liệu nguồn, cấu hình model và commit hash được lưu cùng kết quả.
- Cùng dữ liệu, seed và cấu hình phải tạo cùng split và metric; DL cho phép sai khác số học nhỏ được quy định trong test tolerance.
- Một lệnh pipeline tạo lại bảng kết quả và hình chính của báo cáo.

### 17.4. Dashboard smoke tests

- Cả bốn màn hình mở được với dữ liệu demo.
- Chạy forecast, thay tồn kho và tải CSV không làm ứng dụng lỗi.
- Provenance và fallback warning luôn hiển thị khi có liên quan.

## 18. Cấu trúc mã nguồn dự kiến

```text
configs/                 Cấu hình dữ liệu, feature, model và simulation
data/                    Chỉ chứa README/checksum; dữ liệu lớn không commit
docs/                    Đặc tả, báo cáo và tài liệu sử dụng
src/
  data/                  Ingestion, validation, synthetic generator
  features/              Feature pipeline tránh leakage
  models/                Interface và năm model adapters
  evaluation/            Backtesting, metrics, ablation, reporting
  planning/              BOM, revenue, inventory simulation, ordering
  services/              Forecast orchestration và artifact loading
  app/                   Streamlit pages và components
tests/                   Unit, integration và smoke tests
artifacts/               Model/result sinh ra cục bộ, không commit
```

Mỗi module có một trách nhiệm rõ và giao tiếp qua DataFrame/schema đã kiểm tra. Dashboard không chứa logic feature engineering hoặc lập kế hoạch tồn kho; nó chỉ gọi các service.

## 19. Tiêu chí hoàn thành

Đồ án hoàn thành khi:

1. Pipeline chạy end-to-end bằng một cấu hình hoặc lệnh được tài liệu hóa.
2. Năm mô hình chạy trên cùng cutoff và có bảng WAPE, MAE, RMSSE, bias, thời gian train/inference.
3. Có kết quả ablation và phân tích trường hợp Deep Learning tốt hoặc không tốt.
4. Tập test cuối không được dùng trong quá trình chọn mô hình.
5. Dashboard tạo dự báo 7 ngày, doanh thu, nhu cầu nguyên liệu và danh sách nhập hàng.
6. Inventory simulation so sánh forecast policy với 7-day-average policy bằng cùng điều kiện.
7. Provenance synthetic xuất hiện trên dữ liệu, dashboard, hình và bảng liên quan.
8. Unit/integration/smoke tests cốt lõi đều đạt.
9. Mã nguồn, seed, cấu hình và kết quả chính có thể tái lập.
10. Báo cáo nêu rõ giới hạn và không khẳng định hiệu quả thực tế tại Việt Nam khi chưa có pilot.

Mốc giảm WAPE 5% là mục tiêu nghiên cứu, không phải điều kiện để coi đồ án hoàn thành.

## 20. Kế hoạch 14 tuần

| Tuần | Kết quả bàn giao |
|---|---|
| 1–2 | Tổng quan tài liệu, câu hỏi nghiên cứu, schema và dữ liệu nguồn |
| 3–4 | EDA, data validation, feature pipeline và bộ sinh kịch bản Việt Nam |
| 5–6 | Seasonal Naive, ETS, LightGBM và backtest framework |
| 7–8 | Global LSTM, TFT và thí nghiệm ba seed |
| 9 | Test cuối, ablation, khoảng tin cậy và phân tích sai số |
| 10 | BOM, revenue, inventory simulator và ordering policy |
| 11–12 | Dashboard Streamlit và tích hợp end-to-end |
| 13 | Unit/integration/smoke tests, tái lập và biểu đồ báo cáo |
| 14 | Báo cáo, slide, video/demo và thời gian dự phòng |

Nếu TFT chưa ổn định hết tuần 8, hệ thống vẫn giữ TFT trong báo cáo như một thí nghiệm thất bại có phân tích; phần ứng dụng dùng mô hình tốt nhất còn lại. Không hy sinh backtesting, kiểm thử hoặc dashboard để kéo dài tuning TFT.

## 21. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Biện pháp |
|---|---|
| DL overfit do ít chuỗi | Dùng global model, regularization, early stopping, ba seed và baseline mạnh |
| Synthetic data khiến kết quả quá đẹp | Tách benchmark M5, thêm noise/stockout/drift và không dùng synthetic làm bằng chứng thị trường |
| Leakage từ thời tiết hoặc rolling feature | Lưu `issued_at`, kiểm thử cutoff và fit transformer riêng từng fold |
| Phạm vi quá lớn | Cố định 15 món, tối đa 20 benchmark SKU, 7-day horizon và bốn màn hình |
| TFT tốn thời gian | Giới hạn 12 cấu hình; hết tuần 8 dừng tuning và giữ kết quả hiện có |
| M5 không đại diện quán cà phê | Giới hạn kết luận; đề xuất pilot thật như bước tiếp theo sau đồ án |
| Inventory assumptions không có giá thực | Báo metric vật lý thay vì lợi nhuận giả định; công khai lead time, shelf life và pack size cấu hình |

## 22. Đạo đức, minh bạch và khả năng áp dụng tại Việt Nam

- Không chứa dữ liệu cá nhân hoặc thông tin khách hàng.
- Dữ liệu tổng hợp luôn có provenance và không được gọi là dữ liệu thực.
- Các giả định về công thức, giá, lead time và hạn sử dụng được lưu trong cấu hình, hiển thị trong báo cáo và có thể thay thế.
- Mức độ “áp dụng tại Việt Nam” đến từ schema POS đơn giản, VND, đơn vị metric, menu quen thuộc, lịch Việt Nam, thời tiết TP.HCM, BOM và quy tắc kho của cửa hàng nhỏ.
- Mức độ “đã được kiểm chứng tại Việt Nam” chưa đạt do không có dữ liệu hoặc pilot thật; đây là giới hạn trung tâm, không phải ghi chú phụ.

## 23. Sản phẩm bàn giao

- Mã nguồn Python/PyTorch và cấu hình môi trường.
- Script tải/chuẩn hóa benchmark cùng checksum và hướng dẫn bản quyền dữ liệu.
- Bộ sinh dữ liệu kịch bản Việt Nam và seed tái lập.
- Artifact của các mô hình và bảng kết quả backtesting.
- Inventory simulator và ordering planner.
- Dashboard Streamlit với dữ liệu demo và CSV template.
- Test suite.
- Báo cáo đồ án, slide thuyết trình và kịch bản demo.

## 24. Bước tiếp theo sau đồ án

Bước nâng cấp có giá trị nhất là pilot với một cửa hàng Việt Nam trong tối thiểu 6–12 tháng dữ liệu POS, BOM và tồn kho. Khi đó có thể hiệu chỉnh mô hình, kiểm tra data drift, đo chi phí thật và đánh giá quyết định nhập hàng trong vận hành. Đây là công việc tiếp nối, không thuộc phạm vi đồ án hiện tại.
