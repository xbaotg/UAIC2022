# UIT AI Challenge - Artistic Text Challenge

Source code mô tả cho bài [report](configs/report.pdf) này

- [Chú ý](#WARNING)
- [Dependencies](#dependencies)
- [Xử lý dữ liệu](#PDATA)
  * [Đối với dữ liệu để huấn luyện cho model Recognition](#PDATA-REC)
  * [Đối với dữ liệu để huấn luyện cho model Detection](#PDATA-DET)
- [Tiến hành train model](#TRAIN)
  * [Train model Detection](#TRAIN-DET)
  * [Train model Recognition](#TRAIN-REC)
- [Lưu trained weights và convert thành Inference model](#STORE)
  * [Đối với model Detection (YOLOv7)](#STORE-DET)
  * [Đối với model Recognition (SRN)](#STORE-REC)
  * [Định dạng của thư mục models sau cùng](#STORE-FINAL)
- [Inference](#INFER)
  * [Test model](#INFER-TEST)
  * [Xuất file để nộp](#INFER-SUBMIT)



[](https://user-images.githubusercontent.com/21699486/210822133-1113cbb6-b2eb-4cd0-a558-9b930bcc5ef6.mp4)

<a name="WARNING"></a>
## Chú ý

Nếu như không muốn **Xử lý dữ liệu** và **train model**, ta có thể sử dụng model đã được chúng mình train:

- Model YOLOv7 + SRN: [Drive](https://drive.google.com/file/d/13pkPQT7N7URkuvJwsdz5qpUjTqeWW6QT/view?usp=share_link)

Tiến hành giải nén và đặt vào thư mục gốc (thư mục có chứa các file `predict.py`, `model.py`. Sau đó thực hiện [Inference](#INFER).

> Team có thực hiện thêm việc kiếm tra các ảnh bị đánh nhãn sai, nên số lượng ảnh và quá trình thực hiện có khác đôi chút với hướng dẫn về việc xử lý dữ liệu.

## Cấu trúc folders

```
├── configs
├── data                  - Chứa các ảnh để predict
├── models                
│   ├── det               - Chứa các trained models của YOLOv7
│   └── rec
│       ├── inference     - Chứa các inference models của SRN sau khi export
│       └── train         - Chứa các trained models của SRN
├── PaddleOCR
├── result                - Chứa kết quả sau khi predict
├── tools
│   ├── Detection
│   └── Recognition
└── YOLOv7
```

## Dependencies

```
pip install -r requirements.txt
```

<a name="PDATA"></a>
## Xử lý dữ liệu

<a name="PDATA-REC"></a>
### Đối với dữ liệu để huấn luyện cho model Recognition

1. Đầu tiên ta tiến hành tải bộ dữ liệu đã được đánh nhãn từ BTC [tại đây](https://drive.google.com/file/u/3/d/1NJJA1A8I2Xj5-107E3DFohBNzjWyaaf7/view?usp=share_link)
2. Giải nén và thay đổi tên folder thành `data` và copy vào thư mục `tools/Recognition` với format là:

```
data
├── images 		- chứa các ảnh
├── labels 		- chứa các file json
```
3. Chuyển tới thư mục `tools/Recognition` và thực thi câu lệnh `./process_rec.sh` để tiến hành xử lý các ảnh, sinh thêm các synthetics data, augment data và split data thành tập dữ liệu train và val (tỉ lệ 80-20). Dữ liệu cuối cùng cũng sẽ chính là thư mục `data`.

<a name="PDATA-DET"></a>
### Đối với dữ liệu để huấn luyện cho model Detection

1. Ta thực hiện tương tự như việc xử lý dữ liệu cho model `Recognition` bên trên, nhưng ta cần copy folder `data` vào thư mục `tools/Detection` 
2. Chuyển tới thư mục `tools/Detection` và thực thi câu lệnh `./process_det.sh` để tiến hành xử lý các ảnh, augment data và split data thành tập dữ liệu train và val (tỉ lệ 80-20). Dữ liệu cuối cùng cũng sẽ chính là thư mục `data`.

<a name="TRAIN"></a>
## Tiến hành train model

<a name="TRAIN-DET"></a>
### Train model Detection
Với vấn đề này, Team sử dụng mô hình YOLOv7 được train từ đầu với dữ liệu đã được xử lý bên trên. Để tiến hành train cho model, ta lần lượt thực thi các lệnh bên trong `train_detection.ipynb`.  Trong notebook này, team có ví dụ bằng cách sử dụng [dữ liệu](https://drive.google.com/file/d/1sohRPX_oUXKt6RjwYVR0EJkBJtATmX2o/view) đã được tạo ra và được chúng mình sử dụng.

<a name="TRAIN-REC"></a>
### Train model Recognition
Về việc nhận diện chữ thì chúng mình sử dụng [pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar) mô hình SRN của framework PaddleOCR. Sau đó team mình tiến hành train dựa trên pretrained model đó với tập dữ liệu đã được team tạo ra bên trên. 

Để tiến hành train cho model, ta lần lượt thực thi các lệnh bên trong `train_recognition.ipynb` và cũng trong notebook này, team có ví dụ bằng cách sử dụng [dữ liệu](https://drive.google.com/file/d/1zbmLSW3t7hFq4nd1_GYHqDNHU5ggLKBB/view) đã được tạo ra và được chúng mình sử dụng.

<a name="STORE"></a>
## Lưu trained weights và convert thành Inference model

<a name="STORE-DET"></a>
### Đối với model Detection (YOLOv7)

Sau khi train xong (hoặc không thể train thêm được nữa), ta tiến hành lưu file weights vào thư mục `models/det` và đặt tên file là `yolo.pt`

<a name="STORE-REC"></a>
### Đối với model Recognition (SRN)

- Sau khi việc train hoàn tất (hoặc không thể train thêm được nữa), ta lưu các file weights vào trong thư mục `models/rec/train` với tên là `ocr.pdopt`, `ocr.pdparams`, `ocr.states`.
- Sau đó tiến hành convert thành inference model bằng cách chuyển tới thư mục gốc và chạy lệnh:
```
./export_ocr_inference.sh
```

hoặc 

```
python3 PaddleOCR/tools/export_model.py -c PaddleOCR/configs/rec/rec_r50_fpn_srn.yml -o Global.pretrained_model=models/rec/train/ocr Global.character_dict_path=configs/dict.txt Global.save_inference_dir=models/rec/inference/
```

<a name="STORE-FINAL"></a>
### Định dạng của thư mục models sau cùng
```
models
├── det
│   └── yolo.pt
└── rec
    ├── inference
    │   ├── inference.pdiparams
    │   ├── inference.pdiparams.info
    │   └── inference.pdmodel
    └── train
        ├── ocr.pdopt
        ├── ocr.pdparams
        └── ocr.states
```

<a name="INFER"></a>
## Inference

<a name="INFER-TEST"></a>
### Test model

Để tiến hành test model, ta cho các ảnh vào thư mục `data` sau đó thực hiện lệnh sau để test:

```
python predict.py --task test
```

Các kết quả sẽ được lưu vào trong thư mục `result` với những ảnh giống như thế này

![Image](result/im0005.jpg)

<a name="INFER-SUBMIT"></a>
### Xuất file để nộp

Tương tự như trên, ta cũng cho các ảnh cần nhận diện chữ nghệ thuật vào thư mục `data` và thực hiện lệnh sau để có được các file submit trong thư mục `result`

```
python predict.py --task submit
```
