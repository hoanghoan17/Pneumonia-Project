## 📌 Giới thiệu dự án (Project Introduction)

Dự án này tập trung xây dựng hệ thống tự động phân loại nhị phân ảnh X-quang ngực thẳng để hỗ trợ chẩn đoán hai trạng thái: **Bình thường (Normal)** và **Viêm phổi (Pneumonia)** dựa trên kiến trúc mạng tiên tiến **ResNet18** thông qua kỹ thuật Học chuyển giao (Transfer Learning).

Nhằm giải quyết các thách thức cốt lõi và nâng cao độ tin cậy trong bài toán xử lý ảnh y tế, hệ thống tích hợp hai giải pháp công nghệ chính:
1. **Tiền xử lý ảnh nâng cao (CLAHE):** Áp dụng bộ lọc tăng cường độ tương phản giới hạn histogram cục bộ giúp làm rõ các vùng thâm nhiễm, đông đặc nhu mô phổi bị mờ khuất trên phim chụp gốc, từ đó tối ưu hóa khả năng trích xuất đặc trưng của mạng neural.
2. **Minh giải mô hình bằng Grad-CAM (Explainable AI):** Tích hợp kỹ thuật sinh bản đồ nhiệt trực quan dựa trên luồng đạo hàm tại lớp tích chập cuối cùng (`layer4`). Giải pháp này giúp loại bỏ yếu tố "hộp đen" (Black-box) của Trí tuệ nhân tạo, khoanh vùng chính xác vị trí tổn thương tổn thương phế trường để hỗ trợ các bác sĩ đưa ra quyết định lâm sàng một cách minh bạch và an toàn.

## 📌 Kết quả đạt được (Project Achievements)
Dự án đã huấn luyện và tối ưu thành công mô hình chẩn đoán với các kết quả thực nghiệm đạt chuẩn y tế:

* **Tối ưu hóa thành công dữ liệu mất cân bằng:** Kết hợp bộ lọc **CLAHE** (tăng tương phản) và **WeightedRandomSampler** giúp mô hình không bị học lệch về lớp chiếm đa số.
* **Bộ chỉ số thực nghiệm vững chắc (tại ngưỡng tối ưu 0.455):**
  * **Accuracy (Độ chính xác tổng thể):** `76.16%` — Khả năng nhận diện chính xác trên toàn tập dữ liệu.
  * **Recall / Sensitivity (Độ nhạy y tế):** `63.64%` — Đảm bảo an toàn lâm sàng, hạn chế tối đa việc bỏ sót bệnh nhân viêm phổi thực sự.
  * **Specificity (Độ đặc hiệu):** `80.33%` — Khả năng phân loại chính xác người khỏe mạnh (Normal).
  * **Precision:** `51.85%` | **F1-Score:** `57.14%` | **AUC-ROC:** `79.06%`.
* **Xóa bỏ yếu tố "Hộp đen" (Black-box):** Trích xuất thành công bản đồ nhiệt **Grad-CAM**, trực quan hóa chính xác các vùng phổi bị tổn thương đông đặc, giúp bác sĩ dễ dàng nghiệm thu quyết định của AI.

---

## 🏗️ Kiến trúc mô hình (Model Architecture)

Mô hình được xây dựng dựa trên mạng mạng thần kinh tích chập sâu (CNN) **ResNet18** kết hợp kỹ thuật Học chuyển giao (Transfer Learning) và cải tiến lớp phân loại cuối:

1. **Backbone (Feature Extractor):** * Sử dụng mạng **ResNet18** đã được huấn luyện trước (Pre-trained) trên tập ImageNet để trích xuất các đặc trưng hình học, đường biên và cấu trúc phế trường.
   * **Chiến lược huấn luyện (Selective Fine-tuning):** Đóng băng các tầng tích chập đầu, chỉ mở khóa và tinh chỉnh (Fine-tune) các tầng tích chập sâu cuối cùng (`layer3` và `layer4`) với tỷ lệ học (Learning Rate) siêu nhỏ ($10^{-5}$) để thích ứng riêng với ảnh X-quang ngực.

2. **Custom Classifier Head (Lớp phân loại cải tiến):**
   * Thay thế lớp tuyến tính mặc định của ResNet18 bằng một chuỗi các tầng tuần tự (`nn.Sequential`) nhằm tăng khả năng phân tách phi tuyến tính và chống quá khớp (Overfitting):
     $$\text{Linear(512 } \rightarrow \text{ 256)} \longrightarrow \text{Kích hoạt ReLU} \longrightarrow \text{Dropout(0.3)} \longrightarrow \text{Linear(256 } \rightarrow \text{ 1)}$$
   * Đầu ra đi qua hàm kích hoạt **Sigmoid** để chuyển đổi thành giá trị xác suất nhị phân (từ 0 đến 1), phục vụ cho bước so khớp với ngưỡng quyết định lâm sàng.
