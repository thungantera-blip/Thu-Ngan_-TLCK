# Streamlit Web App: Tối ưu hóa danh mục đầu tư (HOSE 2020-2023)

Ứng dụng web được thiết kế để tối ưu hóa và backtest danh mục đầu tư cổ phiếu trên sàn HOSE dựa trên chiến lược **Bollinger Bands + RSI + Volume Filter**, sử dụng thuật toán **Tối ưu hóa bầy đàn (PSO - Particle Swarm Optimization)**.

## Các tính năng chính

1. **Khám phá dữ liệu**: Xem thông tin tổng quan về bộ dữ liệu sàn HOSE, tra cứu và vẽ biểu đồ lịch sử giá/khối lượng của bất kỳ mã cổ phiếu nào.
2. **Tối ưu hóa tham số (PSO)**: Chạy thuật toán PSO để tự động tìm kiếm bộ tham số tối ưu (BB window, BB std, RSI window, Vol window, RSI lower/upper) giúp tối đa hóa Sharpe ratio trung bình trong giai đoạn học máy (In-Sample).
3. **Kiểm thử danh mục (Backtest)**: Chạy thử nghiệm mô phỏng danh mục đầu tư (Out-of-Sample) từ năm 2021 đến 2023. Hỗ trợ nhiều cơ chế tái cân bằng (Rebalancing) như: hàng tháng, hàng quý, hàng năm và các cơ chế phân bổ tỷ trọng (Equal, Performance).
4. **So sánh hiệu suất**: So sánh trực quan tăng trưởng tài sản (equity curve) của chiến lược với VN-Index, danh mục Buy & Hold rổ cổ phiếu top và danh mục Equal-weight toàn bộ mã.
5. **Kiểm định thống kê**: Tích hợp các kiểm định thống kê T-test và Wilcoxon để đánh giá độ tin cậy của chiến lược.
6. **Biểu đồ kỹ thuật chi tiết**: Vẽ biểu đồ giá kết hợp dải Bollinger Bands, chỉ báo RSI, Khối lượng và các điểm tín hiệu Mua/Bán trực quan cho từng cổ phiếu bằng thư viện tương tác **Plotly**.

---

## Hướng dẫn cài đặt và chạy cục bộ (Local)

### 1. Chuẩn bị môi trường
Yêu cầu đã cài đặt Python (phiên bản 3.9 trở lên).

### 2. Cài đặt các thư viện cần thiết
Mở terminal/command prompt tại thư mục dự án và chạy lệnh sau:
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
Chạy lệnh khởi động Streamlit:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trong trình duyệt của bạn tại địa chỉ `http://localhost:8501`.

---

## Hướng dẫn Deploy lên Streamlit Community Cloud

Để deploy ứng dụng của bạn lên mạng miễn phí thông qua Streamlit, hãy thực hiện các bước sau:

1. **Tạo GitHub Repository**:
   - Đăng nhập vào tài khoản GitHub của bạn.
   - Tạo một repository mới (ví dụ tên: `hose-portfolio-optimization`).
   - Tải toàn bộ mã nguồn lên repository này, bao gồm:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `HOSE_2020_2023.csv` (File dữ liệu cần thiết để app hoạt động)

2. **Deploy trên Streamlit**:
   - Truy cập trang [Streamlit Community Cloud](https://share.streamlit.io/).
   - Đăng nhập bằng tài khoản GitHub của bạn.
   - Click vào nút **"New app"**.
   - Chọn Repository, Branch (thường là `main` hoặc `master`), và File chạy chính (`app.py`).
   - Click nút **"Deploy!"**.

Chờ quá trình cài đặt môi trường hoàn tất (khoảng 1-2 phút), ứng dụng của bạn sẽ hoạt động trực tuyến và bạn có thể chia sẻ đường link cho bất kỳ ai!

---

## Cấu trúc thư mục dự án
```text
├── HOSE_2020_2023.csv       # File dữ liệu cổ phiếu HOSE (2020-2023)
├── app.py                   # Mã nguồn ứng dụng Streamlit chính
├── requirements.txt         # Khai báo các thư viện cần thiết
└── README.md                # Tài liệu hướng dẫn sử dụng (File này)
```
