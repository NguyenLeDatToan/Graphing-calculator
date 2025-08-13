# Graphing Calculator (Python + Colab)

Dự án máy tính vẽ đồ thị với các chức năng:

- Vẽ một hoặc nhiều hàm số trên cùng hệ trục.
- Tạo bảng giá trị (x, y).
- Tô vùng phía trên hoặc phía dưới đường/hàm.
- Giải và vẽ hệ phương trình.
- Phóng to/thu nhỏ (zoom) vùng nhìn.
- Giải phương trình bậc hai và vẽ đồ thị liên quan.

## Dùng trên Google Colab

1. Mở Google Colab: https://colab.research.google.com/
2. File → Upload notebook → chọn `graphing_calculator.ipynb` trong thư mục dự án này (hoặc mở trực tiếp từ GitHub sau khi đẩy repo).
3. Chạy các cell theo thứ tự. Notebook đã có ví dụ và kiểm thử cơ bản.

## Cấu trúc dự án

- `graphing.py`: Thư viện hàm chính để vẽ và xử lý.
- `graphing_calculator.ipynb`: Notebook Colab có ví dụ, UI tham số cơ bản, và kiểm thử.
- `requirements.txt`: Phụ thuộc cần thiết (`sympy`, `matplotlib`, `numpy`).

## Cài đặt cục bộ (tùy chọn)

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -i -c "import graphing as g; print(g.solve_quadratic(1,-3,2))"
```

## Các hàm chính trong `graphing.py`

- `parse_functions(func_strs)` → list sympy expressions từ danh sách chuỗi.
- `plot_functions(func_strs, x_range, shade=None, zoom=None, ax=None)` → vẽ 1+ hàm.
- `table_xy(func_str, x_values)` → bảng (x, y).
- `solve_and_plot_system(eq_strs, x_range, y_range, ax=None)` → nghiệm và đồ thị giao.
- `solve_quadratic(a,b,c)` → nghiệm phương trình bậc hai.

## Gợi ý submission

- Sau khi chạy pass kiểm thử trong Colab, bật chia sẻ: Anyone with the link → Viewer.
- Đẩy dự án lên GitHub rồi gửi link Colab/Repo để chấm.
