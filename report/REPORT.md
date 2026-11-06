# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về cả bức ảnh, vì ảnh có nhiều object nhưng không có thuật toán phân loại từng object, score của rank 1 chỉ nói lên khả năng cả bức ảnh chỉ class đó
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? model pretrain có thể phân loại sẽ trong phạm vi taxonomy và checkpoint, giống như bộ dữ liệu đã được label
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? model được train với checkpoint + taxonomy cụ thể sẽ có thể phân loại ảnh dựa trên class đã học => cần ghi cả id + taxonomy + checkpoint để biết model phân loại dựa trên class của checkpoint nào
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? chọn lớp có mật độ object thuộc lớp đó = cao nhất
6. Vì sao không được lấy `score` của model làm ground truth? vì model không thể tự đánh giá chính nó. nếu ground truth không còn do con người viết thì model luôn có độ chính xác 100% dẫn đến phán đoán thực tế có thể sai lệch

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= cup ; `score`=0.381542 ; `bbox_xyxy`=[119.95,
      272.2,
      141.78,
      304.16] ; `bbox_width`=21.84 ; `bbox_height`=31.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? góc trên trái ở pixel = 119.95, góc dưới phải = 304.16, rộng 21.84 pixel và cao 31.96 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? ngưỡng 0.20 có 17 prediction, ngưỡng 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? check lại prediction. khi threshold giảm sẽ tăng những object bị detect sai 
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). chỉ box những object có thể nhìn được toàn bộ vật thể, không bị mờ, lem hoặc chất lượng ánh sáng kém, box không bị lem, ôm sát vào vật thể và phải cover toàn bộ vật thể. không đoán nhãn vật thể nếu không chắc chắn.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? ưu tiên hỏi lab coach để xác định phương án xử lí phù hợp

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-006 ; `class_name`=bowl ; `score`=0.528201 ; số điểm polygon=`polygon_point_count`=84 ; 3–5 điểm đầu của `polygon_xy`=[
        29.0,
        303.0
      ],
      [
        28.0,
        304.0
      ],
      [
        26.0,
        304.0
      ],
      [
        25.0,
        305.0
      ],
      [
        23.0,
        305.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? box chỉ cho biết vật thể nằm trong box đó, còn polygon cho biết vật thể đó có hình dạng như thế nào dựa vào những tọa độ biên của object
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? instance id chỉ id của 1 vật thể cụ thể trong ảnh, class id chỉ các label(chung), tracking id cũng là chỉ 1 đối tượng cụ thể nhưng xuyên suốt các frame trong 1 video
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo. các điểm phải nằm trong đối tượng, không lem ra ngoài, chú ý các điểm gấp khúc.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? escalation. Vì guideline áp dụng cho các trường hợp phổ thông, rõ ràng. nếu có trường hợp đặc biệt, ưu tiên hỏi coach để xác định đúng phương hướng giải quyết(label theo hướng dẫn của coach hoặc skip)

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | không thể hiện trên ảnh  | ảnh có nhiều vật thể (ảnh traffic có nhiều loại ô tô khác nhau) | gán nhãn ảnh với class có mật độ cao nhất/chỉ gán nhãn ảnh khi có 1 vật thể trong ảnh| label các ảnh có chính xác hay không; xác định label đúng của ảnh trên những ảnh có nhiều object khác nhau |
| Phát hiện vật thể | các vật thể được vẽ bounding box xung quanh, bounding box có class được annotator gán label | vật thể to/nhỏ bất thường(bowl trong ảnh kitchen), vật thể có màu trùng với môi trường(oven trong ảnh kitchen), vật thể bị khuất sau vật khác(các bowl cạnh nhau trong ảnh kitchen) | vẽ bounding box quanh từng vật thể | bounding box có bị lem, chưa cover hết vật thể hay không; label vật thể có chính xác không |
| Instance segmentation | các polygon bám vào biên của từng vật thể | khi có vật thể khác nằm trong instance đang label(dining table có nhiều vật khác nhau ở trên trong ảnh kitchen), các vật nằm cạnh nhau khó xác định biên của vật(các bowl cạnh nhau trong ảnh kitchen) | đặt các điểm biên sát vào mép object, các đường gấp khúc trong object | check các polygon có cover hết vật thể hay không, có bị lem sang các đồ vật khác hay không, check label các instance đã chính xác chưa |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) không ghi thông tin cá nhân vào file .md, không sử dụng các ảnh có bản thân, trường để nhận diện
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? báo cáo cho coach check an toàn trước khi sử dụng dữ liệu để thực hành lab

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
