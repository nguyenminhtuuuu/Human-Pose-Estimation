# Hướng Dẫn Chi Tiết Code & Phân Công Nhóm

## 1. FLOW CHƯƠNG TRÌNH (Inference - Suy Luận)

```
[Input: Ảnh/Video/Webcam]
           ↓
[demo.py: Parse tham số + Tạo model]
           ↓
[Load checkpoint (trọng số đã train)]
           ↓
[For each frame:
   - Resize/Normalize/Pad ảnh
   - Đưa vào MobileNet CNN
   - Lấy output: heatmap + PAF (Part Affinity Field)
   - Extract keypoint từ heatmap
   - Group keypoint thành skeleton người
   - Vẽ skeleton (nhiều màu)
   - Hiển thị
]
           ↓
[Output: Cửa sổ ảnh/video có skeleton vẽ lên]
```

---

## 2. CHI TIẾT CÁC FILE CHÍNH

### A. **demo.py** - Entry Point (Điểm vào chính)
**Nhiệm vụ:** Nhận input, gọi model, hiển thị kết quả.

**Flow chi tiết:**
1. `argparse` (dòng 140-150): Nhận tham số CLI (--checkpoint-path, --video, --images, --cpu).
2. `PoseEstimationWithMobileNet()` (dòng 156): Khởi tạo mạng CNN.
3. `torch.load()` + `load_state()` (dòng 157-158): Load trọng số từ checkpoint vào model.
4. Chọn `ImageReader` hoặc `VideoReader` (dòng 160-163):
   - `ImageReader`: đọc từng ảnh từ danh sách files.
   - `VideoReader`: đọc frame từng frame từ video/webcam.
5. `run_demo()` (dòng 166): Vòng chính xử lý từng frame.

**Các hàm trong demo.py:**
- `infer_fast(net, img, ...)`: Suy luận trên 1 frame:
  - Resize ảnh theo cao độ mạng (256px).
  - Normalize ảnh: trừ mean [128,128,128], chia scale 1/256.
  - Pad ảnh để divisible by stride=8.
  - Forward qua model → lấy heatmap + PAF.
- `run_demo()`: Vòng chính nhận frame → infer → vẽ → hiển thị.

---

### B. **models/with_mobilenet.py** - Kiến Trúc Mạng

**Kiến trúc:** MobileNet (nhẹ, chạy nhanh trên CPU).

**Structure:**
```
Input (ảnh 3 channel)
    ↓
MobileNet backbone (trích xuất feature)
    ↓
Stage 1: Đoán heatmap (19 channel: 18 keypoint + 1 background)
    ↓
Stage 2: Đoán PAF (38 channel: 19 connections × 2 chiều x,y)
    ↓
Output: [heatmap, PAF, ...] từ các stage khác nhau
```

**MobileNet là gì?**
- CNN nhẹ dùng Depthwise Separable Convolution (ít parameter hơn Conv thường).
- Chạy nhanh trên CPU.
- Tradeoff: độ chính xác giảm so với model lớn, nhưng đủ tốt cho real-time.

---

### C. **modules/pose.py** - Lớp Pose & Vẽ Skeleton

**Lớp Pose:** Represent 1 người với 18 keypoint.

**Các thuộc tính chính:**
- `keypoints[18][2]`: Tọa độ (x,y) của 18 điểm (nếu -1 = không tìm thấy).
- `confidence`: Độ tin cậy pose này.
- `bbox`: Khung bounded rectangle (đã tắt vẽ).
- `id`: Tracking id câu người qua frame.
- `filters`: OneEuroFilter làm mượt keypoint qua frame.

**18 keypoints:**
```
0: Nose        | 1: Neck      | 2-4: Right arm    | 5-7: Left arm
8-10: Right leg| 11-13: Left leg | 14: Right eye  | 15: Left eye
16: Right ear  | 17: Left ear
```

**Phương thức `draw()` (dòng 49-68):**
- Loop qua 17 connections (xương).
- Mỗi connection: lấy 2 keypoint đầu cuối.
- Vẽ circle (khớp): màu theo keypoint id từ `kpt_colors[]`.
- Vẽ line (xương): màu theo limb id từ `limb_colors[]`.

**Tracking:** `track_poses()` (dòng 103-129):
- Match pose frame hiện tại vs frame trước.
- Độ similarity: compute overlap keypoint.
- Gán id từ frame trước nếu match.

---

### D. **modules/keypoints.py** - Trích Xuất Keypoint

**`extract_keypoints(heatmap_channel, keypoints_list, offset)`:**
- Input: heatmap của 1 keypoint type (ví dụ "nose").
- Output: Danh sách (x, y, score) các điểm nose tìm được.
- Cách: Tìm local maxima trong heatmap, lọc theo threshold.

**`group_keypoints(all_keypoints_by_type, pafs)`:**
- Input: Tất cả keypoint từng loại + PAF (vector trường).
- Output: Danh sách `pose_entries` → mỗi entry = 1 skeleton người.
- Cách: Dùng PAF để kết nối keypoint lại thành skeleton (bipartite matching).

---

### E. **modules/load_state.py** - Load Checkpoint

**`load_state(model, state_dict)`:**
- Load trọng số từ checkpoint vào model.
- Handle thích ứng nếu số layer không khớp (forward compatible).

---

### F. **modules/one_euro_filter.py** - Smoothing

**OneEuroFilter:** Filter làm mượt keypoint qua frame.
- Input: keypoint thô từ frame hiện tại.
- Output: keypoint được làm mượt.
- Người dùng có thể bật `--smooth 1` để kích hoạt.

---

### G. **val.py** - Hỗ Trợ (Không Dùng Trong Demo)

**Hàm được demo.py import:**
- `normalize(img, mean, scale)`: Chuẩn hóa ảnh (subtract mean, multiply scale).
- `pad_width(img, stride, pad_value, min_dims)`: Pad ảnh cho divisible by stride.

---

## 3. PHÂN CÔNG NHÓM (5 Người)

### **Người 1: Nhóm Trưởng - Tổng Hợp**
**Trách nhiệm:**
- Nắm toàn bộ flow (đã có TEAM_GUIDE.md này).
- Trình bày lên thầy phần tổng quát.
- Phối hợp giữa các thành viên.

**File tìm hiểu:**
- [demo.py](demo.py) (toàn bộ)
- Flow chung

---

### **Người 2: CNN & Model Architecture**
**Trách nhiệm:**
- Giải thích MobileNet là gì, tại sao dùng.
- Kiến trúc mạng (backbone + stages).
- Loss function & training strategy.

**File tìm hiểu:**
- [models/with_mobilenet.py](../models/with_mobilenet.py)
- [modules/conv.py](../modules/conv.py)
- [train.py](../train.py) (nắm ý tưởng, không cần code chi tiết)

**Câu hỏi thầy có thể hỏi:**
- MobileNet khác với CNN thường gì?
- Sao ảnh vào 3 channel, ra 19 + 38 channel?
- Filter làm gì? Depthwise separable convolution là gì?

---

### **Người 3: Input Processing & Inference**
**Trách nhiệm:**
- Giải thích cách đọc input (ảnh/video/webcam).
- Pre-processing (resize, normalize, pad).
- Forward pass qua model.
- Post-processing (extract keypoint, group keypoint).

**File tìm hiểu:**
- [demo.py](demo.py) - `ImageReader`, `VideoReader`, `infer_fast()`, `run_demo()`
- [val.py](../val.py) - `normalize()`, `pad_width()`
- [modules/keypoints.py](../modules/keypoints.py)

**Câu hỏi thầy có thể hỏi:**
- Tại sao phải resize ảnh?
- Mean [128,128,128] từ đâu ra?
- Sai số tính toán có lớn không khi pad?

---

### **Người 4: Keypoint Detection & Skeleton Construction**
**Trách nhiệm:**
- Giải thích heatmap & PAF là gì.
- Cách extract keypoint từ heatmap (local maxima).
- Cách dùng PAF để ghép xương.
- 18 keypoints là những gì.

**File tìm hiểu:**
- [modules/keypoints.py](../modules/keypoints.py) - `extract_keypoints()`, `group_keypoints()`
- [modules/pose.py](../modules/pose.py) - `Pose` class, 18 keypoint names

**Câu hỏi thầy có thể hỏi:**
- Heatmap có bao nhiêu channel? Tại sao (18+1)?
- PAF 38 channel từ đâu?
- Làm sao biết 2 keypoint nào cần nối?

---

### **Người 5: Visualization & Tracking**
**Trách nhiệm:**
- Giải thích cách vẽ skeleton (màu sắc, thickness).
- Tracking người qua frame (id).
- OneEuroFilter (làm mượt).

**File tìm hiểu:**
- [modules/pose.py](../modules/pose.py) - `draw()`, `track_poses()`, colors
- [modules/one_euro_filter.py](../modules/one_euro_filter.py)
- [demo.py](demo.py) - `get_person_color()`

**Câu hỏi thầy có thể hỏi:**
- Tại sao vẽ bằng nhiều màu?
- Tracking id làm sao (match pose qua frame)?
- OneEuroFilter giảm bao nhiêu noise?

---

## 4. CHECKLIST TRẢ LỜI CÂU HỎI CỦA THẦY

### Câu hỏi kỹ thuật chung:
- [ ] Model dùng bao nhiêu parameters?
- [ ] Inference time trên CPU bao lâu?
- [ ] Accuracy (AP) như thế nào?

**Trả lời:**
- MobileNet nhẹ (theo paper: ~6.3M params).
- ~200ms/frame trên CPU i7.
- ~40% AP trên COCO validation set.

### Câu hỏi về hạn chế:
- [ ] Model nhầm gì? False positive/negative?
- [ ] Không nhận diện tốt trường hợp nào (occlusion, etc)?

**Trả lời:**
- Model có thể nhầm khi người bị cắt khung hoặc chồng lấp.
- Không tốt với tư thế cực đoan (yoga).

### Câu hỏi về cải tiến:
- [ ] Có thể cải tiến thế nào? Preprocessing gì?
- [ ] Tại sao không dùng 3D pose?

**Trả lời:**
- Thêm data augmentation, ensemble model.
- 3D có trong repo upstream (link trong README).

---

## 5. THỰC HÀNH TRƯỚC PHÚT VẤN

### Mỗi thành viên chuẩn bị:
1. **Lý thuyết:** Hiểu sâu file của mình.
2. **Code simulation:** Chạy 1 file ví dụ, trace code (F10 step, watch variable).
3. **Trả lời quick tiếng Anh:** Vì thầy có thể hỏi bằng Anh.
4. **Demo live:** Chuẩn bị 2-3 video/ảnh để demo ngay.

### Demo script:
```bash
# 1. Ảnh đơn
python demo.py --checkpoint-path models\checkpoint_iter_370000.pth \
  --images data\human1.jpg --cpu

# 2. Video
python demo.py --checkpoint-path models\checkpoint_iter_370000.pth \
  --video data\videos\sample.mp4 --cpu

# 3. Webcam (nếu có)
python demo.py --checkpoint-path models\checkpoint_iter_370000.pth \
  --video 0 --cpu
```

---

## 6. CẤU TRÚC THUYẾT TRÌNH (Gợi ý)

**Phần 1: Tổng Quan (Nhóm Trưởng - 3 phút)**
- Project là gì? Nhận diện pose người -> skeleton.
- Lợi ích: Realtime, chạy CPU, nhẹ.
- High-level flow: Input → Preprocess → CNN → Extract Keypoint → Draw → Output.

**Phần 2: Chi Tiết Kỹ Thuật (Các thành viên - 10 phút)**
- P2.1 (Người 2): CNN architecture (2 phút).
- P2.2 (Người 3): Input pre/post-processing (2 phút).
- P2.3 (Người 4): Keypoint & Skeleton (2 phút).
- P2.4 (Người 5): Visualization & Tracking (2 phút).

**Phần 3: Demo (Nhóm Trưởng + Tất Cả - 3 phút)**
- Chạy demo ảnh, video, webcam.
- Giải thích output realtime.

**Phần 4: Q&A - Đáp Lời Thầy (Tất Cả - 4 phút)**
- Chuẩn bị câu trả lời trước.

---

## 7. THỨ TỰ ĐỌC FILE HỢP LÝ

Nếu chưa biết gì, đọc theo thứ tự này:

1. Đọc README.md (overview).
2. Đọc demo.py (flow chính).
3. Đọc from_mobilenet.py (model).
4. Đọc modules/pose.py (visualization).
5. Đọc modules/keypoints.py (processing).
6. Đọc train.py (optional, hiểu training strategy).

---

## 8. GHI CHÚ QUAN TRỌNG

- **Không train:** Project này chỉ chạy inference (dùng model đã train sẵn).
- **No augmentation at inference:** Augmentation chỉ dùng lúc training.
- **Upstream attribution:** Giữ link tác giả gốc (Daniil Osokin, Apache 2.0 License).
- **Git setup:** Remote upstream = tác giả, origin = team repo mới.

---

Chúc nhóm thuyết trình tốt! 🎉
