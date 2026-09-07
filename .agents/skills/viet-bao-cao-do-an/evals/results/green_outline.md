# Dàn ý báo cáo đồ án CNTT (theo yêu cầu Khoa)

> **Tình trạng thông tin:** Chưa có tên đề tài, dữ liệu, công nghệ hay kết quả thực nghiệm cụ thể. Vì vậy, các mục dưới đây là khung điền chỗ trống. Không được biến placeholder thành số liệu hoặc kết luận chính thức nếu chưa có bằng chứng.

## Phần đầu

1. Bìa chính và bìa phụ theo mẫu của Khoa: tên đề tài, sinh viên, MSSV, GVHD, lớp và năm học (`[CẦN BỔ SUNG: ...]`).
2. Lời cảm ơn.
3. Nhận xét và chữ ký của giảng viên hướng dẫn.
4. Mục lục (cập nhật trường mục lục trước khi nộp).
5. Danh mục hình vẽ.
6. Danh mục bảng biểu.
7. Danh mục từ viết tắt và thuật ngữ.

## Chương 1. Giới thiệu (khoảng 10%)

### 1.1. Bối cảnh và vấn đề

- Mô tả hiện trạng của **[đối tượng/đơn vị sử dụng]**, vấn đề cần giải quyết và người bị ảnh hưởng.
- Nêu dẫn chứng có thể kiểm tra; nếu chưa có nguồn, ghi `[CẦN NGUỒN]` thay vì tự tạo con số.
- Xác định ranh giới: “Đồ án làm **X**, không làm **Y**”.

### 1.2. Mục tiêu

Phát biểu từ 3 đến 5 mục tiêu đo được. Có thể dùng bảng sau để theo dõi việc đối chiếu ở Chương 5.

**Bảng 1.1. Mục tiêu và tiêu chí hoàn thành**

| Mã | Mục tiêu | Chỉ số/tiêu chí chấp nhận | Cách đo và nguồn bằng chứng | Trạng thái |
|---|---|---|---|---|
| MT-01 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [log/bảng đo/biên bản] | Chưa xác minh |
| MT-02 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [log/bảng đo/biên bản] | Chưa xác minh |
| MT-03 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [log/bảng đo/biên bản] | Chưa xác minh |

### 1.3. Đối tượng và phạm vi

- Đối tượng, môi trường sử dụng, dữ liệu và khoảng thời gian: `[CẦN BỔ SUNG: ...]`.
- Chức năng, nền tảng và người dùng nằm trong phạm vi.
- Danh sách phần không làm, giả định và giới hạn truy cập dữ liệu.

### 1.4. Phương pháp thực hiện

Tóm tắt chu trình khảo sát → phân tích → thiết kế → hiện thực → kiểm thử → đánh giá. Tên mô hình, tham số, phiên bản thư viện và cách chia dữ liệu chỉ ghi sau khi đã xác minh từ mã nguồn hoặc nhật ký chạy.

### 1.5. Bố cục báo cáo

Một đoạn ngắn giới thiệu nội dung của năm chương, tài liệu tham khảo và phụ lục.

## Chương 2. Cơ sở lý thuyết và công trình liên quan (khoảng 20%)

### 2.1. Khái niệm và kiến thức được sử dụng

Chỉ giữ các khái niệm, thuật toán, mô hình, giao thức hoặc công nghệ xuất hiện trong thiết kế và hiện thực. Mỗi ý tưởng hoặc số liệu kế thừa phải có nguồn IEEE đã xác minh (`[CẦN NGUỒN]` nếu chưa có).

### 2.2. Khảo sát công trình/sản phẩm tương tự

Chọn khoảng 5 đến 10 công trình hoặc sản phẩm thật. Với mỗi mục, nêu bài toán, phương pháp, dữ liệu, ưu điểm và hạn chế; không điền tên tác giả, năm, DOI hay URL khi chưa kiểm chứng.

**Bảng 2.1. So sánh các công trình/giải pháp liên quan**

| Mục | Nguồn IEEE | Bài toán/dữ liệu | Phương pháp | Ưu điểm | Hạn chế | Liên hệ với đồ án |
|---|---|---|---|---|---|---|
| CT-01 | [CẦN NGUỒN] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] |
| CT-02 | [CẦN NGUỒN] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] |

### 2.3. Khoảng trống và định hướng

Tổng hợp hạn chế còn lại của các giải pháp, chỉ ra khoảng trống mà đồ án lấp vào và tiêu chí dùng để đánh giá khoảng trống đó.

### 2.4. Lý do chọn công nghệ

So sánh các phương án theo tiêu chí phù hợp (dữ liệu, chi phí, khả năng triển khai, thời gian đáp ứng, giấy phép). Lý do phải dựa trên bằng chứng hoặc ràng buộc của đồ án, không viết “vì đã quen dùng”.

## Chương 3. Phân tích và thiết kế hệ thống (khoảng 25%)

### 3.1. Khảo sát hiện trạng và yêu cầu

- Actor hoặc nhóm người dùng, mục tiêu và bối cảnh sử dụng.
- Yêu cầu chức năng (F-01, F-02, …) và phi chức năng (hiệu năng, bảo mật, khả dụng, khả năng bảo trì).
- Use case hoặc user story kèm tiền điều kiện, luồng chính, luồng ngoại lệ và điều kiện kết thúc.

**Hình 3.1. Sơ đồ ngữ cảnh và các actor**  
Đặt sơ đồ sau một đoạn văn giới thiệu; bổ sung `[CẦN BỔ SUNG: sơ đồ]` khi chưa có.

### 3.2. Kiến trúc và luồng dữ liệu

- Kiến trúc tổng thể, các thành phần và giao tiếp giữa chúng.
- Luồng dữ liệu từ đầu vào đến lưu trữ, xử lý và đầu ra.

**Hình 3.2. Kiến trúc tổng thể của hệ thống**  
Giải thích vai trò từng thành phần và các quyết định thiết kế ngay trước hình.

### 3.3. Thiết kế dữ liệu

- ERD hoặc lược đồ dữ liệu; mô tả bảng, khóa, ràng buộc và quy tắc toàn vẹn.
- Với đề tài AI: nguồn dữ liệu, tiêu chí chọn mẫu, tiền xử lý, chia tập và nguy cơ rò rỉ dữ liệu.

**Bảng 3.1. Mô tả các bảng dữ liệu**

| Bảng/nguồn | Trường chính | Kiểu dữ liệu | Ràng buộc/ý nghĩa | Nguồn |
|---|---|---|---|---|
| [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [nội bộ/IEEE] |

### 3.4. Thiết kế xử lý

- Sơ đồ hoạt động hoặc tuần tự cho các kịch bản quan trọng.
- Công thức, tham số và giả định; công thức đánh số (3.1), (3.2) và giải thích ký hiệu.

**Hình 3.3. Sơ đồ tuần tự của kịch bản [CẦN BỔ SUNG]**

### 3.5. Thiết kế giao diện

Wireframe hoặc ảnh mẫu cho màn hình chính, trạng thái lỗi và phản hồi người dùng. Nêu lý do bố trí dựa trên yêu cầu, không chỉ liệt kê ảnh.

## Chương 4. Hiện thực và kiểm thử (khoảng 25%)

### 4.1. Môi trường và phiên bản

Ghi phần cứng, hệ điều hành, ngôn ngữ, framework, thư viện, cơ sở dữ liệu và phiên bản lấy từ tệp cấu hình hoặc lệnh kiểm tra. Mục còn thiếu đánh dấu `[CẦN BỔ SUNG]`.

### 4.2. Cấu trúc mã nguồn và mô-đun

Đưa cây thư mục và mô tả trách nhiệm của mô-đun. Chỉ trích 10 đến 20 dòng xử lý cốt lõi trong chương; mã dài đưa xuống phụ lục.

**Đoạn mã 4.1. Hàm [CẦN BỔ SUNG]**  
`[CẦN BỔ SUNG: đoạn mã đã kiểm tra và giấy phép nếu kế thừa]`

### 4.3. Chức năng và kịch bản sử dụng

Mỗi chức năng có ảnh chụp rõ nét, dữ liệu đầu vào mẫu, các bước thao tác và đầu ra quan sát được.

**Hình 4.1. Màn hình [CẦN BỔ SUNG]**

### 4.4. Kiểm thử

Lập ca kiểm thử với đủ bốn cột bắt buộc: đầu vào, kết quả mong đợi, kết quả thực tế và đạt/không đạt. Không ghi “đạt” nếu chưa chạy và lưu bằng chứng.

**Bảng 4.1. Ma trận ca kiểm thử chức năng**

| Mã ca | Đầu vào | Kết quả mong đợi | Kết quả thực tế | Đạt/Không đạt |
|---|---|---|---|---|
| TC-01 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | Chưa chạy |
| TC-02 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | Chưa chạy |
| TC-03 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | Chưa chạy |

Bao gồm tối thiểu các nhóm: đầu vào hợp lệ; thiếu hoặc sai định dạng; biên và dữ liệu rỗng; quyền truy cập; lỗi phụ thuộc bên ngoài; luồng tích hợp giữa các mô-đun. Nếu là mô hình dự báo/AI, thêm baseline, tập kiểm tra theo thời gian, MAE/RMSE/MAPE hoặc chỉ số đã thống nhất, thời gian suy luận và phân tích lỗi theo nhóm dữ liệu.

### 4.5. Kiểm thử đơn vị và tích hợp

Nêu phạm vi, công cụ, số ca đã chạy, môi trường và log liên quan. Kết quả chỉ được báo cáo sau khi có tệp log hoặc bảng đo có thể tái lập.

## Chương 5. Kết quả, đánh giá và kết luận (khoảng 15%)

### 5.1. Đối chiếu mục tiêu

**Bảng 5.1. Đối chiếu mục tiêu và bằng chứng**

| Mã mục tiêu | Tiêu chí ở Bảng 1.1 | Kết quả đo | Bằng chứng | Đạt/Đạt một phần/Chưa đạt |
|---|---|---|---|---|
| MT-01 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | Chưa xác minh |
| MT-02 | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | [CẦN BỔ SUNG] | Chưa xác minh |

### 5.2. Số liệu và phản hồi

Trình bày các chỉ số đã đo, đơn vị, điều kiện đo, khoảng tin cậy (nếu có), phản hồi người dùng thử và cách lấy mẫu. Không dùng các tính từ như “rất tốt” hoặc “khá nhanh” thay cho số liệu.

### 5.3. Phân tích lỗi

Nêu trường hợp hệ thống cho kết quả sai, tần suất, nguyên nhân có bằng chứng và ảnh hưởng đến mục tiêu. Phân biệt lỗi đã quan sát với giả thuyết cần kiểm tra.

### 5.4. Hạn chế và hướng phát triển

Thừa nhận giới hạn dữ liệu, phạm vi, hiệu năng, bảo mật hoặc khả năng triển khai. Mỗi hướng phát triển cần nêu việc cụ thể, điều kiện cần và cách đánh giá; không viết đề xuất như kết quả đã đạt.

### 5.5. Kết luận

Tóm tắt đóng góp thực tế trong phạm vi đã kiểm chứng, mức đạt của từng mục tiêu và câu hỏi còn bỏ ngỏ.

## Tài liệu tham khảo

- Dùng chuẩn IEEE, đánh số theo thứ tự xuất hiện `[1]`, `[2]`, … trong nội dung.
- Danh mục chỉ chứa nguồn thực sự được trích dẫn; mỗi `[n]` trong bài phải có đúng một mục và ngược lại.
- Ưu tiên bài báo, sách, tiêu chuẩn và tài liệu chính thức. Không bịa tác giả, năm, DOI, URL; mục chưa xác minh ghi `[CẦN NGUỒN]` cho tới khi được kiểm tra.
- Phân biệt phần kế thừa, mã nguồn thư viện và đóng góp của đồ án; ghi giấy phép khi cần.

## Phụ lục

1. Mã nguồn tiêu biểu hoặc đường dẫn kho mã cùng giấy phép.
2. Bảng dữ liệu mẫu, mô tả schema, quy trình tiền xử lý và cấu hình thí nghiệm.
3. Toàn bộ ma trận ca kiểm thử, log, ảnh chụp và kết quả đo.
4. Hướng dẫn cài đặt, chạy, tài khoản mẫu và cách tái lập demo.
5. Hình ảnh giao diện đầy đủ hoặc video demo nếu không phù hợp đưa vào chương chính.

## Quy tắc trình bày và tự kiểm tra

- Khổ A4; Times New Roman cỡ 13; giãn dòng 1,5; đoạn 6 pt; lề trái 3 cm, phải 2 cm, trên 2 cm, dưới 2 cm. Phần đầu đánh số La Mã, nội dung dùng số Ả Rập.
- Tối đa ba cấp tiêu đề (1.1, 1.1.1). Mỗi chương bắt đầu trang mới.
- Bảng đặt tên phía trên, hình đặt chú thích phía dưới; đánh số theo chương (Bảng 3.1, Hình 4.1). Công thức đánh số bên phải; mã dùng phông đơn cách.
- Mọi hình, bảng, công thức và đoạn mã phải được giới thiệu trong văn bản trước khi xuất hiện, có nguồn nếu lấy từ bên ngoài và có trong danh mục tương ứng.
- Tìm chuỗi `--`, sáo ngữ và khẳng định tuyệt đối để sửa theo ngữ cảnh. Giữ dấu nối `-` và gạch ngang `–` đúng nghĩa, không thay máy móc thành `--`.
- Kiểm tra chính tả, cập nhật mục lục/danh mục, đối chiếu trích dẫn hai chiều, chạy lại ca kiểm thử trên môi trường demo và kiểm tra trùng lặp theo quy định của Khoa. Không hứa hẹn hoặc tìm cách né công cụ Turnitin; nội dung phải dựa trên nguồn và trải nghiệm thực tế của người viết.
