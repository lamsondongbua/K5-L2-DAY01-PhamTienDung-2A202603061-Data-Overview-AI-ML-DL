# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-07

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

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

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    - Record này nói về cả bức ảnh. Vì các lí do:
        + "task": "image_classification"  (sử dụng để phân loại toàn ảnh) 
        + "model_file" : "yolo11n-cls.pt" (hậu tố -cls = classification model, khác với -det (detection))
        + "image_width" và "image_height" : "640" và "428" (kích thước toàn bộ ảnh, không có crop/region)
        + Nếu là vật thể riêng, bắt buộc phải có bbox, x, y,... (tọa độ của bounding box)

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    - Danh sách tên lớp (class list) đến từ file .pt khi load model (vì file .pt của YOLO không chỉ chứa trọng số mà còn chứa cả metadata đi kèm)
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    - Nếu chỉ lưu mỗi `class_name` thì sẽ xảy ra vấn đề:
        + Vấn đề 1: cùng `class_name`, khác `taxonomy_name` -> khác nghĩa (không biết class_name thuộc taxonomy nào thì không biết model đang nói đến gì)
        + Vấn đề 2: cùng `taxonomy_name`, cùng `class_id` -> nhãn tên có thể thay đổi theo version (dữ liệu ghi lúc trước và lúc sau khi Ultralytics đổi nhãn không khớp nhau khi query)
        + Vấn đề 3: cùng `class_name`, khác `taxonomy_name` -> `class_id` khác nhau (`class_id` vô nghĩa khi không có `taxonomy_name`).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    - Định nghĩa "chủ thể chính" (guideline nêu rõ tiêu chí)
    - Quy tắc khi nhiều vật ngang nhau (guideline phải có tie-breaking rule)
    - Phân biệt nhãn cảnh với nhãn vật thể 
    - Ngưỡng tự tin tối thiểu
    - Ví dụ minh họa cụ thể
6. Vì sao không được lấy `score` của model làm ground truth?
    - `score`: là đầu ra của model, không phải sự thật của ảnh.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= bowl ; `score`= 0.381215 ; `bbox_xyxy`=[0.32, 305.02, 87.69, 371.32] ; `bbox_width`= 87.37; `bbox_height`= 66.3
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    - Góc trên-trái với tọa độ (0.32, 305.02) : cách mép trái ảnh ~0px, cách mép trên ảnh ~305px
    - Góc dưới-phải với tọa độ (87.69, 371.32) : cách mép trái ảnh ~88px, cách mép trên ảnh ~371px
    - Rộng 87.37px
    - Cao 66.30px

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    - Chọn hai ngưỡng 0.35 và 0.60. Số prediction của từng ngưỡng:
        + Ngưỡng 0.35: 11 prediction
        + Ngưỡng 0.60: 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    - Phân loại từng box
    - Kiểm tra box có bị trùng lặp hay không
    - Kiểm tra bbox có khớp vật thể không
    - Quyết định accept / reject / sửa cho từng box
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    - Quy tắc : Box phải bao sát phần nhìn thấy được của vật thể
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    - Vật bị cắt mép : oven 
    - Em sẽ viết thành quy tắc trong guideline, vì dựa theo tiêu chí và đánh giá sau: 
        + Có xảy ra thường xuyên không -> ảnh thực tế luôn có vật ở rìa
        + Cách xử lý có nhất quán không -> luôn bao phần nhìn thấy, dừng lại ở mép ảnh 
        + Cần chuyên môn đặc biệt không -> không (bất kì annotator nào cũng áp dụng được)
        + Có thể viết thành quy tắc rõ ràng không -> có

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-002; `class_name`= bowl ; `score`= 0.735744; số điểm polygon=`polygon_point_count`= 67 ; 3–5 điểm đầu của `polygon_xy`= [53, 344], [52, 345], [50, 345], [49, 346], [48, 346]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    - Polygon cho thêm thông tin về: 
        + Hình dạng thực (tròn, bầu dục)
        + Phần nào là vật, phần nào là nền
        + Vật bị cắt/che ở phần nào 
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    - `instance_id` dùng để định danh duy nhất cho một vật thể cụ thể trong một ảnh.
    - Câu "Nó không phải class ID và không phải tracking ID" được hiểu: 
        + class ID: trả lời "đây là loại gì"
        + tracking ID: trả lời "cái này ở frame sau có phải cùng vật không?"
        + instance ID: trả lời "đây là cái cụ thể nào?" 
    => Vì vậy chúng khác nhau, mỗi trường xác định một khác 
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    - Quy tắc vẽ biên mask: Vẽ đường biên sát theo đường viền ngoài cùng nhìn thấy được của vật thể — không bao vùng nền, không cắt vào thịt vật thể.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Instance được chỉ ra: kitchen-001 chạm vào vùng bếp/counter : Phần dưới thân người (y ≈ 300–348) bị che bởi mặt bàn bếp — chỉ thấy phần trên, phần thân dưới khuất sau counter.
    - Em sẽ dùng guideline, vì: Tình huống này xảy ra thường xuyên trong ảnh thực tế (người đứng sau bàn, quầy, xe...) và cách xử lý nhất quán, có thể mô tả thành quy tắc rõ ràng (Vẽ polygon bao phần thân người nhìn thấy được, dừng tại đường giao với vật che — không ước đoán phần bị khuất phía sau.)

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh (class_id, class_name, taxonomy_name) | Ảnh traffic có nhiều xe buýt và ô tô; top-1 cab chỉ 0.510915, minibus 0.164284, police_van 0.085848 nên khó chọn một nhãn | Chọn một lớp từ taxonomy theo tiêu chí guideline (vật lớn nhất, gần giữa nhất) | Nhãn có nằm trong taxonomy và đúng tiêu chí không; không dùng score làm căn cứ |
| Phát hiện vật thể | Một hộp bbox_xyxy (pixel) và một lớp cho mỗi vật | Ảnh kitchen: person 0.61 bị cắt ở mép ảnh; ngưỡng 0.20 cho 17 hộp nhưng 0.60 chỉ 6 hộp, nhiều hộp yếu (spoon, bottle) | Vẽ hộp ôm sát phần nhìn thấy của từng vật | Hộp có sát không, có thiếu hoặc thừa vật, có sai lớp không |
| Instance segmentation | Một polygon polygon_xy và một instance_id cho mỗi vật | Ảnh kitchen: dining table 0.58 chồng lên bowl 0.70 và 0.74; person 0.43 bị che; traffic-001 (bus) có 120 điểm polygon nên vẽ tay tốn công | Vẽ biên theo vật, tách riêng từng instance | Biên mask có khớp vật không, các instance chồng nhau có bị lẫn không |

Nguồn số liệu: output của notebook (classification, detection kèm so sánh ngưỡng 0.20 / 0.35 / 0.60 với 17 / 11 / 6 vật, segmentation) và các ảnh trong `day1_lab_outputs/visuals/`.

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    - Chỉ dùng ba ảnh COCO công khai (traffic, kitchen, dining) do notebook tải về; không tải ảnh cá nhân hay dữ liệu nội bộ, và không ghi họ tên, MSSV, email hay số điện thoại vào báo cáo.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    - Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, tôi dừng lại và báo Lab Coach (mentor) thay vì tự xử lý.

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
