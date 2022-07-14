# CS331- Trích xuất thông tin hóa đơn

## Phát hiện hóa đơn (Yolov4): 
Đây là bước đầu trong cả pipeline bài toán. Bước này có chức năng là xác định được vị trí hóa đơn nằm ở đâu trong ảnh và cắt vùng ảnh chỉ chứa hóa đơn ra.

Link training collab: [link collab](https://colab.research.google.com/drive/1LFXgVfauf-XOLrBzIraGPPrcY996WxHQ?usp=sharing)

Link data (đã được label và được xử lý thủ công cho thẳng lại): [data labeld](https://l.facebook.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1HYpdcZ4c47bNg82GcP3UsYdcVXPybWAf%2Fview%3Fusp%3Dsharing%26fbclid%3DIwAR3vz1r2GflT4C1rl4yMiEVC5a7lmUh2jWRrBbFYnOz7AaYxzUm5mji4_ww&h=AT2TvkSWXErM2UyEw9bA3V92Qpxt8yEeMt4hSiByk1LnM9LTX_V0P6rsqDqksEtjSKpTyeYfNH3rEhLQ7i2cgbgXq51BxqGbjMFlpxWOjJilMPIHjWB8tyQkOeGXPikSeuVcnA)

### Sử dụng: 

```
git clone 
cd CS331_R-CIPT
pip3 install --upgrade pip
pip3 install -r /Yolov4_receipt_detect/setup.txt
```
```
python3 Yolov4_infer.py 
```

File Yolov4_infer.py có input là link ảnh cần được xử lý và output là ảnh đã được cropped.

**Note**: Muốn chạy được file Yolov4_infer.py theo đúng mục đích thì cần phải chỉnh lại path của model: **model_path** (có thể tải [tại đây](https://drive.google.com/drive/u/1/folders/1y2ZUnXhe3ZADboXt53YiAJ23BfLUFhBV)). Hoặc nếu muốn train lại thì chúng tôi cũng cung cấp data mà chúng tôi đã label sẵn [tại đây](https://l.facebook.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1HYpdcZ4c47bNg82GcP3UsYdcVXPybWAf%2Fview%3Fusp%3Dsharing%26fbclid%3DIwAR3vz1r2GflT4C1rl4yMiEVC5a7lmUh2jWRrBbFYnOz7AaYxzUm5mji4_ww&h=AT2TvkSWXErM2UyEw9bA3V92Qpxt8yEeMt4hSiByk1LnM9LTX_V0P6rsqDqksEtjSKpTyeYfNH3rEhLQ7i2cgbgXq51BxqGbjMFlpxWOjJilMPIHjWB8tyQkOeGXPikSeuVcnA)

## Phát hiện văn bản (Paddle): 

**Cài đặt**
```
cd text_detector/PaddleOCR
pip3 install -e .
python3 -m pip install paddlepaddle==2.0rc1 -i https://mirror.baidu.com/pypi/simple
cd CS331_R-CIPT
```
### Sử dụng: 
Bước này sẽ tìm vị trí của vùng chữ trên ảnh. Chúng tôi sử dụng pre-trained từ PaddleOCR mà không finetune lại gì cả. Các bạn hãy download pre-trained từ [link này](https://drive.google.com/drive/u/1/folders/172-JTWrqQcoKm0bqjU4pJ6KcbTA0z7Y4), giải nén và chỉnh sửa đường dẫn đến pre-trained trong file **paddle_infer.py**

File **paddle_infer.py** có input là ảnh đã được croped từ bước trước và output sẽ là file txt chứa các bounding box mà nó dự đoán. File txt này sẽ được lưu lại ở det_out_txt_dir mà ta đã gán trong file này.

**Note**: trong file **paddle_infer.py** các bạn cần chỉnh sửa lại **det_model_dir** là đường dẫn của model vừa được tải về. Ngoài ra các bạn còn chỉnh thêm vị trí mà các bạn muốn kết quả trả về thông qua chỉnh lại **det_out_txt_dir**. 

## Nhận diện văn bản (VietOCR):

**Cài đặt**
```
cd text_classifier/vietocr
pip3 install -e .
cd CS331_R-CIPT
```

Đây chính là bước OCR đọc chữ từ vùng ảnh đã được detect từ bước trên. Theo như tác giả nói rằng là mô hình khá nhạy cảm với sự thay đổi nhỏ của ảnh đầu vào khi sử dụng pretrained model trên tập dữ liệu mới chưa được huấn luyện. Do đó chúng mình đã huấn luyện cả 2 dạng model **vgg_transformer** và **vgg_seq2seq**.

### Training VietOCR:

Phần này dành cho các bạn muốn training lại VietOCR trên bộ dữ liệu mới. Ở đây để bộ dữ liệu phục vụ sát với domain mà chúng tôi đang làm nhất nên chúng tôi sẽ cắt các vùng ảnh chứa text từ các ảnh sẵn có (đã được xử lý thủ công và crop) ví dụ như [ở đây của chúng tôi](https://drive.google.com/file/u/1/d/1dyQt3PXT1wXtzWRgf-UtrxkSNhkuIrYQ/view?usp=sharing). Tuy nhiên chúng tôi sẽ không cắt các vùng chữ 1 cách thủ công mà sẽ tận dụng lại phần text detection của Paddle để xác định vị trí chữ trong ảnh sau đó nhờ vào thông tin này chúng tôi tiến hành cắt ảnh vùng chữ ra. Việc còn lại của chúng ta đó là label chữ cho từng ảnh được cắt ra này.

**Cài đặt**
```
cd train_vietocr
python3 train_vietocr/splitfile.py 
```

**Note**: Để thực hiện cắt nhiều ảnh vùng chữ cùng 1 lúc và trên toàn bộ data ảnh đúng các bạn cần chỉnh sửa đường dẫn tại các biến như sau:  **images_dir** (thư mục chứa ảnh đã được crop [có thể tham khảo tại đây](https://drive.google.com/file/u/1/d/1dyQt3PXT1wXtzWRgf-UtrxkSNhkuIrYQ/view?usp=sharing), txt_dir (thư mục chứa các file txt tương ứng với ảnh hóa đơn ở **images_dir**), **output_ocr**="./ocr_data" (thư mục này sẽ lưu data phục vụ cho VietOCR sau khi chạy file này).


Sau khi đã có được data và label xong. Chúng ta tiến hành training. 
**Cài đặt**
```
python3 train_vietocr/trainocr.py 
cd CS331_R-CIPT
```

**Note**: Gỉa sử đã có được các file ảnh text và label chữ cho các file này. Chúng ta cần phải chia train/test/val, chúng tôi có cung cấp hàm **split_mcocr()** để phục vụ cho việc này ở ngay trong file **trainocr.py**. Các bạn có thể comment phần training để mình chạy phần chia data trước rồi training sau. Ngoài ra các bạn nên xem thêm cách training và điều chỉnh file config ngay tại repo gốc của tác giả [tại đây](https://github.com/pbcquoc/vietocr). File model sau khi train xong sẽ được lưu ở **train_vietocr/weights**.

### Sử dụng: 

```
python3 vietocr_infer.py
```

**Note**

Sau khi training xong chúng ta sẽ có được file model ở **train_vietocr/weights**. Việc của bạn bây giờ sẽ là chọn file model (.pth) và file cfg tương ứng, rồi sau đó tiến hành chỉnh sửa trong file **text_classifier/vietocr/vietocr/vietocr_class.py**. 

File **vietocr_infer.py** sẽ nhận input là ảnh hóa đơn (có từ bước 1) và file txt các vùng chữ trong hóa đơn (có từ bước 2). Để thực hiện các bạn cần điều chỉnh lại đường dẫn thông qua 2 biến **img_path và anno_path** trong file này.

## key information extraction:

Bước này sử dụng PICK model của tác giả wenwenyu để trích xuất thông tin từ hóa đơn. 

Link model đã được training sẵn cho bài này: https://drive.google.com/drive/u/1/folders/10sCCINIrFx3MjU2OGlIHGbcFi8t8-JfH

### Sử dụng: 

```
python3 Pick_infer.py
```

Để chạy được file **Pick_infer.py**, các bạn cần chỉnh lại file các **args** tương ứng với các file mà các bạn muốn thao tác lên trong file **PICK-pytorch/test.py** trước tiên. 
