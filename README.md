# UIT AI Challenge - Artistic Text Challenge

## Dependencies
Tiến hành cài đặt các dependencies trước có thể chạy
```
pip install -r requirements.txt
```

## Xử lý dữ liệu
### Đối với dữ liệu để huấn luyện cho model Recognition

1. Đầu tiên ta tiến hành tải bộ dữ liệu đã được đánh nhãn từ BTC [tại đây](https://drive.google.com/file/u/3/d/1NJJA1A8I2Xj5-107E3DFohBNzjWyaaf7/view?usp=share_link)
2. Giải nén và thay đổi tên folder thành `data` và copy vào thư mục `tools/Recognition` với format là:
```
data
├── images 		- chứa các ảnh
├── labels 		- chứa các file json
```
3. Thực thi câu lệnh `./process_rec.sh` để tiến hành xử lý các ảnh, sinh thêm các synthetics data, augment data và split data thành tập dữ liệu train và val (tỉ lệ 80-20)

> Team có thực hiện thêm việc kiếm tra các ảnh bị đánh nhãn sai, nên số lượng ảnh và quá trình thực hiện có khác đôi chút với hướng dẫn bên trên.
