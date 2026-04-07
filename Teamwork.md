**Ý nghĩa từng file**

1. coco.py: Dataset loader cho train/val, tạo heatmap keypoint và PAF map từ annotation, đồng thời tạo mask học.
2. transformations.py: Data augmentation và chuyển đổi keypoint format (scale, rotate, crop-pad, flip, convert keypoints).
3. with_mobilenet.py: Định nghĩa kiến trúc mạng chính (backbone MobileNet + CPM + initial/refinement stages).
4. conv.py: Các block convolution tái sử dụng (conv thường, depthwise, depthwise không BN).
5. get_parameters.py: Gom nhóm tham số theo loại layer để set optimizer với LR/weight_decay khác nhau.
6. keypoints.py: Hậu xử lý từ heatmap + PAF để trích keypoint và ghép thành từng người.
7. load_state.py: Load checkpoint linh hoạt, hỗ trợ load từ checkpoint chuẩn hoặc từ backbone MobileNet.
8. loss.py: Hàm loss L2 có mask cho heatmap/PAF.
9. one_euro_filter.py: Bộ lọc One Euro để làm mượt tọa độ keypoint theo thời gian.
10. pose.py: Đối tượng Pose, vẽ skeleton, gán ID, tracking giữa frame và smoothing.
11. convert_to_onnx.py: Export model PyTorch sang ONNX để deploy.
12. make_val_subset.py: Tạo tập val con để test nhanh.
13. prepare_train_labels.py: Chuyển COCO json sang định dạng nội bộ pickle để train.
14. demo.py: Chạy demo ảnh/video/webcam, infer nhanh, vẽ skeleton, tracking ID.
15. TRAIN-ON-CUSTOM-DATASET.md: Hướng dẫn sửa code khi train dataset custom (đổi keypoint schema, số output channel, mapping).
16. train.py: Pipeline huấn luyện đầy đủ: dataloader, augment, optimizer groups, scheduler, checkpoint, val định kỳ.
17. val.py: Validation theo COCO metric, infer đa tỉ lệ, convert output về format COCO để chấm AP.

**Phần bắt buộc chung cho cả nhóm**
14. demo.py: phải hiểu luồng infer ảnh/video, hậu xử lý, vẽ pose, track id.
15. TRAIN-ON-CUSTOM-DATASET.md: phải hiểu toàn bộ các điểm cần sửa khi đổi dataset/keypoint schema.

**Phần chuyên sâu từng người**
**Quỳnh Anh**: datasets/coco.py, datasets/transformations.py, scripts/prepare_train_labels.py 
**Hồng Yến**: models/with_mobilenet.py, modules/conv.py 
**Nhật Nguyên**: train.py, modules/get_parameters.py, modules/loss.py, modules/load_state.py 
**Minh Tú**: modules/keypoints.py, modules/pose.py, modules/one_euro_filter.py
**Kim Phụng**: val.py, scripts/convert_to_onnx.py, scripts/make_val_subset.py



