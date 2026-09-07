---
name: viet-bao-cao-do-an
description: Use when writing, outlining, revising, or reviewing a Vietnamese information-technology graduation/project report that must follow yeu-cau-bao-cao.md, including its five-chapter structure, IEEE citations, academic formatting, testing evidence, natural prose, and plagiarism-safe source handling.
---

# Viết báo cáo đồ án CNTT

## Mục tiêu

Tạo bản báo cáo có thể kiểm chứng từ dữ liệu thật của đồ án. Báo cáo phải giúp người đọc hiểu vấn đề, cách làm, bằng chứng và giới hạn; không biến phần còn thiếu thành “sự thật” cho đủ câu chữ. Đây là hỗ trợ biên tập và diễn đạt, không phải cách né Turnitin hay che giấu việc dùng AI.

## Quy trình trước khi viết

1. Đọc `yeu-cau-bao-cao.md` trong workspace nếu có. Nếu không có, dùng chuẩn tóm tắt ở mục “Khung báo cáo” bên dưới và hỏi người dùng quy định riêng của Khoa.
2. Lập bảng dữ liệu đầu vào: tên đề tài, bối cảnh, mục tiêu đo được, đối tượng/phạm vi, dữ liệu, phương pháp, công nghệ và phiên bản, kết quả đã chạy, hạn chế, tài liệu nguồn. Ghi nguồn của từng con số và quyết định.
3. Tách ba trạng thái thông tin: **đã cung cấp/đã xác minh**, **suy luận cần người dùng duyệt**, và **chưa có**. Với trạng thái cuối, đặt `[CẦN BỔ SUNG: ...]`; không tự điền.
4. Xác nhận đầu ra cần viết (dàn ý, một chương, toàn quyển, hay chỉnh sửa) và mức độ hoàn thành thực tế. Nếu người dùng yêu cầu viết toàn bộ nhưng thiếu bằng chứng, tạo bản nháp có chỗ trống và danh sách việc cần hoàn thiện.

## Khung báo cáo

Giữ bố cục sau, trừ khi biểu mẫu chính thức của Khoa yêu cầu khác:

- Phần đầu: bìa, lời cảm ơn, nhận xét GVHD, mục lục, danh mục hình/bảng/từ viết tắt.
- Chương 1 – Giới thiệu (~10%): bối cảnh và vấn đề; mục tiêu 3–5 mục tiêu đo được; đối tượng và phạm vi (nêu cả phần không làm); phương pháp tóm tắt; bố cục báo cáo.
- Chương 2 – Cơ sở lý thuyết và công trình liên quan (~20%): chỉ trình bày kiến thức được dùng lại; khảo sát 5–10 công trình/sản phẩm; bảng so sánh; khoảng trống; lý do chọn công nghệ có căn cứ.
- Chương 3 – Phân tích và thiết kế (~25%): yêu cầu chức năng/phi chức năng, actor và use case/user story; kiến trúc và luồng dữ liệu; ERD/lược đồ hoặc mô tả dữ liệu và tiền xử lý; thiết kế xử lý, giao diện.
- Chương 4 – Hiện thực và kiểm thử (~25%): môi trường và phiên bản; cấu trúc mã nguồn/mô-đun; chức năng kèm kịch bản; ca kiểm thử đầu vào–mong đợi–thực tế–đạt; kiểm thử đơn vị/tích hợp.
- Chương 5 – Kết quả, đánh giá và kết luận (~15%): đối chiếu từng mục tiêu (đạt/một phần/chưa đạt); số liệu và phản hồi; phân tích lỗi; hạn chế và hướng phát triển cụ thể.
- Cuối quyển: tài liệu tham khảo, phụ lục mã nguồn/dữ liệu/hướng dẫn cài đặt.

Mọi hình, bảng, công thức và đoạn mã phải được giới thiệu trong văn bản trước khi xuất hiện, đánh số theo chương (ví dụ Hình 4.1, Bảng 3.2, (3.1)), có chú thích đúng vị trí và có trong danh mục tương ứng.

## Quy tắc nội dung và bằng chứng

- Không khẳng định mô hình, độ chính xác, thời gian, số giao dịch, tác động kinh doanh hoặc trạng thái “đã hoàn thành” nếu chưa có log, bảng đo hoặc xác nhận của người dùng. Khi chỉ có kế hoạch, dùng “dự kiến” và đặt tiêu chí đánh giá.
- Không bịa tài liệu, tác giả, DOI, URL, năm xuất bản hay trích dẫn. Chỉ đưa nguồn đã cung cấp hoặc đã xác minh; nếu thiếu, ghi `[CẦN NGUỒN]` và đề nghị người dùng bổ sung.
- Dùng IEEE: đánh số theo thứ tự xuất hiện `[1]`, `[2]`; danh mục chỉ gồm nguồn thực sự được trích dẫn. Kiểm tra hai chiều: mỗi `[n]` có mục tài liệu và mỗi mục tài liệu được gọi trong bài.
- Phân biệt phần kế thừa, mã nguồn thư viện và đóng góp của đồ án; ghi giấy phép/nguồn khi cần. Diễn đạt lại bằng cấu trúc và lập luận của đồ án, nhưng vẫn trích dẫn ý tưởng hoặc số liệu gốc. Không hứa hẹn tỷ lệ Turnitin; nhắc người dùng kiểm tra và khai báo AI theo quy định của Khoa.
- Đề tài phải nói rõ “làm X, không làm Y”, có dữ liệu khả dụng, điểm mới và demo được. Không mở rộng phạm vi chỉ để làm báo cáo dài.

## Văn phong tự nhiên, chuẩn học thuật

- Dùng ngôi khách quan phù hợp: “hệ thống được xây dựng…”, nhưng giữ thuật ngữ và cách gọi nhất quán với đồ án. Câu thường một ý; thay tính từ mơ hồ (“rất tốt”, “khá nhanh”, “tối ưu”) bằng số đo và điều kiện đo.
- Gắn nhận xét với bối cảnh cụ thể của dữ liệu, mô-đun hoặc ca kiểm thử. Trộn độ dài câu vừa phải, dùng liên từ tự nhiên, tránh mở đoạn lặp như “Trong bối cảnh chuyển đổi số hiện nay”, “Có thể thấy rằng”, “đóng vai trò vô cùng quan trọng”, “khẳng định tính hiệu quả”.
- Không mô phỏng lỗi để “trông giống người” và không cố né bộ phát hiện AI. Ưu tiên bản thảo trung thực, sau đó để người dùng bổ sung trải nghiệm, quyết định và số liệu của chính họ.
- Giữ dấu câu đúng nghĩa: dấu nối `-` cho từ ghép/khoảng mã khi cần; dấu gạch ngang tiếng Việt `–` cho khoảng hoặc chen ý theo quy ước; không tự động đổi một dấu `-` thành `--`. Dấu `--` chỉ xuất hiện trong mã/lệnh hoặc khi nguồn gốc bắt buộc.

## Kiểm tra trước khi giao

Chạy checklist này và báo rõ mục nào chưa thể kiểm tra:

- [ ] Năm chương và tỷ trọng/nội dung phù hợp; Chương 5 đối chiếu đúng mục tiêu Chương 1.
- [ ] Không còn số liệu, kết quả, nguồn hoặc tính năng do trợ lý tự bịa; mọi ô thiếu có `[CẦN BỔ SUNG]`.
- [ ] Bảng ca kiểm thử có đủ bốn cột; sơ đồ/hình/bảng/mã có số, chú thích, nguồn (nếu có) và được nhắc trong văn bản.
- [ ] Trích dẫn IEEE khớp hai chiều; không có nguồn “trang trí”.
- [ ] Tìm các chuỗi `--`, sáo ngữ và khẳng định tuyệt đối; sửa theo ngữ cảnh, không thay máy móc.
- [ ] Thuật ngữ, tên mô hình, đơn vị, phiên bản và cách viết hoa nhất quán; số liệu có đơn vị và điều kiện đo.
- [ ] Nêu hạn chế thật và hướng phát triển có thể thực hiện; không biến đề xuất thành kết quả đã đạt.
- [ ] Nhắc người dùng cập nhật mục lục/danh mục trong Word, soát PDF, kiểm tra chính tả và đối chiếu quy định mới nhất của Khoa trước khi nộp.

## Khi thông tin chưa đủ

Hỏi tối đa các câu hỏi quyết định chất lượng: đề tài và đối tượng; mục tiêu/phạm vi; dữ liệu và thời gian; phương pháp/công nghệ; kết quả đo; nguồn bắt buộc; mẫu định dạng của Khoa. Có thể cung cấp một đoạn **minh họa**, nhưng phải gắn nhãn “Ví dụ – không dùng làm số liệu nộp” và không để lẫn vào bản chính.

## Mẫu đầu ra tối thiểu

```text
Phần đang viết: [chương/mục]
Dữ liệu đã xác minh: ...
Chỗ cần bổ sung: [CẦN BỔ SUNG: ...]

## 1.x Tiêu đề
[Đoạn văn gắn với đề tài và bằng chứng]

Bảng/Hình x.y: [tên]
[nội dung hoặc chỗ đặt]
Nguồn: [nếu dùng nguồn ngoài]

Tự kiểm tra: [mục đã đạt] | [mục còn thiếu]
```

## Các lỗi cần dừng lại

| Dấu hiệu | Cách xử lý |
|---|---|
| “Cứ tự thêm số liệu/nguồn cho đủ” | Dừng, hỏi hoặc đánh dấu `[CẦN BỔ SUNG]`. |
| “Viết Chương 5 trước dù chưa chạy mô hình” | Chỉ viết khung đánh giá, không viết kết quả như sự thật. |
| “Đổi hết `-` thành `--` để khó bị phát hiện AI” | Giữ dấu theo ngữ nghĩa và quy chuẩn tiếng Việt. |
| “Paraphrase là không cần trích dẫn” | Vẫn trích nguồn của ý tưởng/số liệu; viết lập luận riêng. |
| “Thêm lý thuyết để tăng số trang” | Chỉ giữ kiến thức được dùng trong thiết kế/hiện thực/đánh giá. |
