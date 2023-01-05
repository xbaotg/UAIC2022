# Xử lý dữ liệu cho việc train Recognition
---
Quá trình này sẽ gồm các bước:

- Sinh ra các chữ đã được crop từ tập đã được đánh nhãn
- Sinh ra các Synthetics Text
- Gộp 2 thư mục ảnh sinh ra và gộp 2 file labels.txt lại
- Thực hiện Augment các ảnh trên
- Tiến hành split các ảnh thành tập train và val (tỉ lệ 80-20)

---
Chuẩn bị thư mục data theo format sau:

- data
    - images: chứa các ảnh
    - labels: chứa các file label dạng json

Sau đó chạy lệnh sau để sinh ra các ảnh.
`./process_rec.sh`

Và thư mục sau khi chạy xong chính là chính thư mục `data`
---
**Cái này không phản ảnh đúng việc Team đã làm, vì Team có thực hiện thêm việc lọc ảnh bị label lỗi trước khi thực hiện
các việc như sinh Synthetics Text và Augment data.**
