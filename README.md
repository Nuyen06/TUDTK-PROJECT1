# Part 3: Giải Hệ Phương Trình Tuyến Tính và Phân Tích Hiệu Năng

## Tổng Quan

Phần 3 tập trung vào ba phương pháp giải hệ phương trình tuyến tính Ax = b:
1. **Gaussian Elimination (từ Part 1)** - Phương pháp khử Gauss có chọn pivot
2. **Cholesky Decomposition (từ Part 2)** - Phân rã cho ma trận Symmetric Positive Definite (SPD)
3. **Gauss-Seidel** - Phương pháp lặp

Ngoài ra, phần này cũng phân tích:
- **Hiệu năng (Benchmark)**: So sánh thời gian chạy và độ chính xác giữa các phương pháp
- **Ổn định số học (Numerical Stability)**: Tác động của condition number đến sai số
- **Trực quan hóa**: Vẽ đồ thị log-log để so sánh độ phức tạp

---

## Cấu Trúc File

```
part3/
├── solvers.py              # Các hàm solver chính
├── benchmark.py            # Hàm benchmark và phân tích hiệu năng
├── analysis.ipynb          # Jupyter Notebook cho báo cáo thực nghiệm
└── README.md               # File này
```

---

## 1. File: `solvers.py`

File này chứa các hàm số chính để giải hệ phương trình tuyến tính và các hàm hỗ trợ.

### Khai báo Kiểu Dữ Liệu
```python
Matrix = List[List[float]]  # Ma trận 2 chiều
Vector = List[float]        # Vector 1 chiều
EPS = 1e-12                 # Ngưỡng epsilon để xử lý sai số số học
```

### Hàm Tải Module từ Part 1
#### `_load_part1_gaussian_eliminate() -> Callable`
- **Mục đích**: Nạp hàm `gaussian_eliminate` từ Part 1 một cách động
- **Chi tiết**: 
  - Xây dựng đường dẫn đến thư mục part1
  - Sử dụng `importlib.import_module` để tái sử dụng code Part 1
  - Đảm bảo Part 3 dùng đúng phiên bản Gaussian đã hoàn thiện
- **Trả về**: Hàm `gaussian_eliminate` từ Part 1

### Hàm Hỗ Trợ Cơ Bản

#### `copy_matrix(A: Matrix) -> Matrix`
- **Mục đích**: Sao chép sâu (deep copy) ma trận
- **Lý do**: Tránh sửa đổi trực tiếp dữ liệu đầu vào
- **Phương pháp**: Sao chép từng hàng `[row[:] for row in A]`

#### `validate_system(A: Matrix, b: Vector) -> None`
- **Mục đích**: Kiểm tra tính hợp lệ của hệ Ax = b
- **Kiểm tra**:
  - Ma trận A không rỗng
  - A phải là ma trận vuông (n × n)
  - Tất cả hàng của A phải có cùng số cột
  - Vector b phải có độ dài bằng số hàng của A
- **Ngoại lệ**: Ném `ValueError` nếu dữ liệu không hợp lệ

#### `is_symmetric(A: Matrix, tol: float = 1e-10) -> bool`
- **Mục đích**: Kiểm tra ma trận đối xứng (yêu cầu cho Cholesky)
- **Phương pháp**: Kiểm tra điều kiện A[i][j] ≈ A[j][i] cho i < j
- **Tối ưu**: Chỉ kiểm tra nửa trên ma trận để giảm phép so sánh

#### `norm2(v: Vector) -> float`
- **Mục đích**: Tính chuẩn Euclid (chuẩn 2) của vector
- **Công thức**: ||v|| = √(Σ v_i²)
- **Ứng dụng**: Dùng trong tính residual và kiểm tra hội tụ

#### `mat_vec_mul(A: Matrix, x: Vector) -> Vector`
- **Mục đích**: Nhân ma trận với vector
- **Công thức**: (Ax)_i = Σ_j A[i][j] × x[j]
- **Ứng dụng**: Tính Ax để đánh giá sai số residual

#### `relative_residual(A: Matrix, x: Vector, b: Vector, eps: float = EPS) -> float`
- **Mục đích**: Tính sai số residual tương đối
- **Công thức**: ||Ax - b|| / ||b||
- **Ý nghĩa**: Đánh giá độ chính xác của nghiệm x so với hệ Ax = b
- **Trường hợp đặc biệt**: Nếu ||b|| ≈ 0, trả về ||Ax - b||

### Hàm Thế Tiến - Thế Lùi (Substitution)

#### `forward_substitution(L: Matrix, b: Vector, eps: float = EPS) -> Vector`
- **Mục đích**: Giải hệ Ly = b (L là ma trận tam giác dưới)
- **Phương pháp**: Thế tiến từ trên xuống
  - Tính y[0] từ L[0][0] × y[0] = b[0]
  - Tính y[i] từ (b[i] - Σ_{j<i} L[i][j] × y[j]) / L[i][i]
- **Độ phức tạp**: O(n²)

#### `back_substitution(U: Matrix, b: Vector, eps: float = EPS) -> Vector`
- **Mục đích**: Giải hệ Ux = b (U là ma trận tam giác trên)
- **Phương pháp**: Thế lùi từ dưới lên
  - Tính x[n-1] từ U[n-1][n-1] × x[n-1] = b[n-1]
  - Tính x[i] từ (b[i] - Σ_{j>i} U[i][j] × x[j]) / U[i][i]
- **Độ phức tạp**: O(n²)

### Solver Gaussian Elimination

#### `gaussian_solve_pp(A: Matrix, b: Vector, eps: float = EPS) -> Tuple[Vector, Dict]`
- **Mục đích**: Giải hệ Ax = b bằng khử Gauss có chọn pivot cục bộ (Partial Pivoting)
- **Bước thuật toán**:
  1. **Kiểm tra đầu vào** và tạo ma trận tăng cường [A|b]
  2. **Khử Gauss với pivot**:
     - Với mỗi cột k, chọn hàng có trị tuyệt đối lớn nhất tại vị trí [i][k]
     - Hoán đổi hàng nếu cần
     - Khử phần tử dưới pivot bằng phép biến đổi hàng
  3. **Thế ngược**: Sử dụng `back_substitution` để lấy nghiệm x
- **Trả về**: (x, {"swaps": số lần hoán đổi hàng})
- **Ưu điểm**: Giảm sai số số học so với phương pháp không chọn pivot

#### `gaussian_solve_part1(A: Matrix, b: Vector) -> Tuple[Vector, Dict]`
- **Mục đích**: Gọi trực tiếp solver Gaussian từ Part 1 và quy đổi kết quả
- **Quy trình**:
  1. Kiểm tra hệ bằng `validate_system`
  2. Nạp và gọi `gaussian_eliminate` từ Part 1
  3. Chuyển đổi kết quả từ Fraction sang float
  4. Xử lý các trường hợp vô nghiệm hoặc vô số nghiệm
- **Trả về**: (x, {"swaps": số lần hoán đổi, "source": "part1.gaussian_eliminate"})
- **Lợi ích**: Đảm bảo tính nhất quán giữa các phần nhất toàn của đề án

### Solver Cholesky Decomposition

#### `cholesky_decomposition(A: Matrix, eps: float = EPS) -> Matrix`
- **Mục đích**: Phân rã ma trận SPD A thành A = LL^T
- **Điều kiện**:
  - A phải là ma trận vuông
  - A phải đối xứng
  - A phải xác định dương (positive definite)
- **Bước thuật toán**:
  1. Tính từng cột j của L:
     - L[j][j] = √(A[j][j] - Σ_{k<j} L[j][k]²)
     - L[i][j] = (A[i][j] - Σ_{k<j} L[i][k]L[j][k]) / L[j][j] (i > j)
- **Độ phức tạp**: O(n³/3) - nhanh hơn Gaussian thông thường
- **Trả về**: Ma trận L (phần dưới tam giác)

#### `solve_cholesky(A: Matrix, b: Vector, eps: float = EPS) -> Tuple[Vector, Dict]`
- **Mục đích**: Giải hệ Ax = b bằng phân rã Cholesky
- **Bước thuật toán**:
  1. **Phân rã**: A = LL^T bằng `cholesky_decomposition`
  2. **Thế tiến**: Giải Ly = b bằng `forward_substitution`
  3. **Thế lùi**: Giải L^T x = y bằng `back_substitution`
- **Trả về**: (x, {"factorization": "cholesky"})
- **Ưu điểm**: Chỉ dùng được cho ma trận SPD nhưng nhanh hơn Gaussian

### Solver Gauss-Seidel (Phương Pháp Lặp)

#### `is_strictly_row_diagonally_dominant(A: Matrix) -> bool`
- **Mục đích**: Kiểm tra điều kiện hội tụ của Gauss-Seidel
- **Điều kiện**: |a_ii| > Σ_{j≠i} |a_ij| với mọi hàng i
- **Ý nghĩa**: Nếu đúng, Gauss-Seidel đảm bảo hội tụ

#### `gauss_seidel(A: Matrix, b: Vector, x0: Optional[Vector] = None, tol: float = 1e-8, max_iter: int = 10000, eps: float = EPS) -> Tuple[Vector, Dict]`
- **Mục đích**: Giải hệ Ax = b bằng phương pháp lặp Gauss-Seidel
- **Bước thuật toán**:
  1. **Khởi tạo**: x = x0 hoặc vector 0 nếu x0 không cho
  2. **Vòng lặp**: Với mỗi lần lặp k từ 1 đến max_iter:
     - Với mỗi biến i: x[i] = (b[i] - Σ_{j<i} a_ij × x[j] - Σ_{j>i} a_ij × x_old[j]) / a_ii
     - Sử dụng "giá trị mới" x[j] nếu j < i, "giá trị cũ" x_old[j] nếu j > i
  3. **Điều kiện dừng**: max|x[i] - x_old[i]| < tol
- **Trả về**: (x, {"iterations": số vòng lặp thực hoạt, "converged": có hội tụ hay không, "diag_dominant": có chéo trội hay không})
- **Độ phức tạp mỗi vòng**: O(n²), nhưng ít lần lặp hơn phương pháp Jacobi
- **Ưu điểm**: Phương pháp lặp, dùng được cho hệ lớn; x[i] dùng giá trị mới còn x_old[j] dùng giá trị cũ giúp hội tụ nhanh hơn Jacobi

### Main Function (Test nhanh)
```python
if __name__ == "__main__":
```
- Dùng ma trận SPD kiểm tra: A = [[4,4,2], [4,8,6], [2,6,9]], b = [14, 26, 23]
- Chạy cả 3 solver: Gaussian, Cholesky, Gauss-Seidel
- In nghiệm và residual để đối chiếu

---

## 2. File: `benchmark.py`

File này chứa các hàm để benchmark hiệu năng và phân tích ổn định số của các solver.

### Hàm Tạo Dữ Liệu Test

#### `make_spd_matrix(n: int, seed: int) -> Matrix`
- **Mục đích**: Tạo ma trận SPD ngẫu nhiên với kích thước n×n
- **Công thức**: A = R^T R + nI
  - R là ma trận ngẫu nhiên trong [-1, 1]
  - R^T R luôn bán xác định dương
  - Cộng nI đảm bảo A xác định dương rõ ràng (eigenvalues > n)
- **Ý nghĩa seed**: Cho phép tái lập kết quả
- **Tính chất**: Condition number κ(A) ≈ n/1 (có thể kiểm soát được)

#### `make_random_vector(n: int, seed: int) -> Vector`
- **Mục đích**: Sinh vector b ngẫu nhiên trong [-1, 1]
- **Lý do**: Tránh thiên lệch (bias) một chiều trong test

#### `mean(values: List[float]) -> float`
- **Mục đích**: Tính trung bình cộng
- **Trường hợp đặc biệt**: Trả về `nan` nếu danh sách rỗng

### Hàm Benchmark

#### `time_one_run(method_name: str, A: Matrix, b: Vector) -> Tuple[float, float, Dict]`
- **Mục đích**: Đo thời gian chạy 1 lần và đánh giá sai số
- **Phương pháp hỗ trợ**:
  - `"gauss_part1"`: Gaussian từ Part 1
  - `"cholesky"`: Phân rã Cholesky
  - `"gauss_seidel"`: Phương pháp lặp Gauss-Seidel
- **Trả về**: (thời gian (giây), residual tương đối, thông tin phương pháp)
- **Chi tiết**: Dùng `time.perf_counter()` để đo thời gian chính xác

#### `benchmark_sizes(sizes: List[int], repeats: int = 5, base_seed: int = 123) -> Dict[str, Dict[int, Dict[str, float]]]`
- **Mục đích**: Benchmark toàn bộ với nhiều kích thước và nhiều lần lặp
- **Quy trình**:
  1. Cố định 3 phương pháp: "gauss_part1", "cholesky", "gauss_seidel"
  2. Với mỗi kích thước n trong sizes:
     - Tạo ma trận SPD và vector b cố định (để so sánh công bằng)
     - Với mỗi phương pháp, chạy `repeats` lần và lấy trung bình
  3. Thu thập các thông tin:
     - `time_avg`: Thời gian trung bình (giây)
     - `residual_avg`: Residual trung bình
     - `converged_ratio`: Tỷ lệ hội tụ (cho Gauss-Seidel)
     - `iter_avg`: Số vòng lặp trung bình (cho Gauss-Seidel)
- **Trả về**: Cấu trúc lồng {phương pháp: {n: {metric: giá trị}}}
- **Output**: In từng kết quả để theo dõi tiến độ

#### `plot_runtime_loglog(results: Dict[str, Dict[int, Dict[str, float]]], sizes: List[int]) -> None`
- **Mục đích**: Vẽ đồ thị log-log so sánh thời gian chạy
- **Chi tiết**:
  - Trục x (log): Kích thước ma trận n
  - Trục y (log): Thời gian chạy (giây)
  - Vẽ 3 đường cho 3 phương pháp
  - Thêm đường tham chiếu O(n³) để so sánh độ phức tạp
- **Ý nghĩa**: Trên log-log, đường O(n³) là thẳng, giúp nhận biết xu hướng tăng

#### `print_result_table(results: Dict[str, Dict[int, Dict[str, float]]], sizes: List[int]) -> None`
- **Mục đích**: In bảng tổng hợp kết quả benchmark
- **Định dạng**: Bảng dễ đọc với các cột:
  - Kích thước n
  - 3 phương pháp với thời gian, residual, tỷ lệ hội tụ, số vòng lặp

### Hàm Phân Tích Ổn Định Số

#### `condition_number_2(A: Matrix) -> float`
- **Mục đích**: Tính số điều kiện κ(A) theo chuẩn 2
- **Công thức**: κ(A) = σ_max / σ_min (tỷ số eigenvalue lớn nhất / nhỏ nhất)
- **Dùng**: NumPy `np.linalg.cond(A, 2)` để ước lượng (chỉ dùng cho phân tích, không phải cài đặt)
- **Ý nghĩa**: 
  - κ(A) ≈ 1: Ma trận điều kiện tốt, sai số nhỏ
  - κ(A) >> 1: Ma trận điều kiện xấu, sai số có thể lớn

#### `hilbert_matrix(n: int) -> Matrix`
- **Mục đích**: Tạo ma trận Hilbert - ví dụ kinh điển điều kiện kém
- **Công thức**: H[i][j] = 1/(i + j + 1)
- **Tính chất**: Danh tiếng về condition number rất lớn (tăng mũ theo n)
- **Ứng dụng**: So sánh với SPD để minh chứng tác động của condition number

#### `make_spd_matrix_fixed(n: int, seed: int = 0) -> Matrix`
- **Mục đích**: Wrapper rõ ý nghĩa của `make_spd_matrix` trong case study

#### `stability_case_study(ns: List[int], seed: int = 0) -> Dict[str, List[Dict[str, float]]]`
- **Mục đích**: Chạy case study so sánh Hilbert vs SPD
- **Quy trình**:
  1. Dùng b = [1, 1, ..., 1] để giữ bài toán đơn giản
  2. Với mỗi kích thước n:
     - Tạo ma trận Hilbert và SPD
     - Chạy 3 solver và thu thập:
       - Condition number κ(A)
       - Residual của mỗi solver
       - Tỷ lệ hội tụ của Gauss-Seidel
  3. Báo cáo minh chứng: κ lớn → sai số lớn?
- **Trả về**: {"hilbert": [...], "spd": [...]}

#### `print_stability_report(report: Dict[str, List[Dict[str, float]]]) -> None`
- **Mục đích**: In báo cáo ổn định số dưới dạng bảng
- **Cột**: n, κ(A), residual_gauss, residual_cholesky, residual_gs, gs_converged
- **Định dạng**: Chuẩn hóa số để xử lý nan/inf

### Main Function
```python
if __name__ == "__main__":
```
- **Quy trình chính**:
  1. Chạy benchmark với sizes = [50, 100, 200, 500, 1000], repeats = 5
  2. In bảng tổng hợp
  3. Vẽ đồ thị log-log
  4. Chạy case study ổn định số với Hilbert vs SPD
  5. In báo cáo ổn định số

---

## 3. File: `analysis.ipynb`

Jupyter Notebook để trình bày kết quả thực nghiệm và báo cáo.

### Cấu Trúc Notebook

#### Cell 1: Markdown - Giới thiệu
- Tiêu đề: "Analysis Notebook - Phần 3"
- Giải thích mục tiêu: So sánh 3 solver (Gauss Part 1, Cholesky, Gauss-Seidel)
- Nêu rõ các phần cần test: tốc độ, residual, ổn định số

#### Cell 2: Code - Import Module
```python
from solvers import (gaussian_solve_part1, solve_cholesky, gauss_seidel, relative_residual)
from benchmark import (benchmark_sizes, plot_runtime_loglog, ...)
```
- Kiểm tra nạp thành công, gán `MODULES_READY` 
- Cấu hình matplotlib `figure.figsize = (9, 6)`

#### Cell 3: Markdown - "1. Kiểm tra nhanh độ đúng"
- Giải thích: Dùng bộ dữ liệu SPD đã cập nhật từ Part 2

#### Cell 4: Code - Kiểm tra nhanh
- Dùng ma trận SPD: A = [[4,4,2], [4,8,6], [2,6,9]], b = [14, 26, 23]
- Chạy 3 solver, in nghiệm và residual
- Mục đích: Xác nhận 3 phương pháp cho cùng kết quả trên bộ test SPD

#### Cell 5-6: Markdown + Code - Benchmark chính
- Chạy `benchmark_sizes` với sizes = [50, 100, 200, 500, 1000]
- In bảng tổng hợp bằng `print_result_table`

#### Cell 7-8: Markdown + Code - Đồ thị log-log
- Gọi `plot_runtime_loglog` để vẽ runtime vs n trên thang logarithmic
- So sánh với đường tham chiếu O(n³)

#### Cell 9: Markdown - "4. Phân tích ổn định số học"
- Nêu lý thuyết condition number κ(A)
- Dự kiến case study Hilbert vs SPD

#### Cell 10: Code - Case study ổn định số
- Chạy `stability_case_study` với ns = [5, 8, 10]
- In báo cáo bằng `print_stability_report`
- So sánh κ(A) và residual giữa Hilbert (điều kiện xấu) vs SPD (điều kiện tốt)

#### Cell 11: Markdown - "5. Kết luận"
- Tổng hợp kết quả:
  - **Độ đúng**: 3 solver cho cùng nghiệm trên bộ SPD
  - **Tốc độ**: Gauss-Seidel nhanh nhất, Cholesky nhanh hơn Gauss
  - **Ổn định**: Gauss & Cholesky có residual nhỏ (10^-16), GS tốt hơn trên SPD
  - **Hội tụ GS**: Hội tụ với SPD, không hội tụ với Hilbert
  - **Độ phức tạp**: Gauss & Cholesky tăng cỡ O(n³), GS tăng chậm hơn

---

## 4. File: `project_spec_extracted.txt`

### 1. Test Nhanh
```bash
python solvers.py
```
Chạy test cơ bản với bộ dữ liệu SPD kiểm tra.

### 2. Chạy Benchmark Đầy Đủ
```bash
python benchmark.py
```
Chạy benchmark toàn bộ với 3 kích thước [50, 100, 200, 500, 1000] và in báo cáo.

### 3. Xem Kết Quả trong Notebook
```
Bấm Ctrl+Shift+D để mở Jupyter Notebook
Chạy từng cell để xem kết quả thực nghiệm
```

---

## Data flow

```
input: Ma trận A (SQLight trừ Cholesky) + Vector b

Part 1 Gaussian ──┬──> residual
                 │
Cholesky ────────┼──> residual
                 │
Gauss-Seidel ────┼──> residual
                 │
benchmark.py ────┴──> Thời gian + Residual + Stability
                      │
                      └──> analysis.ipynb (vẽ đồ thị & báo cáo)
```

---

## Ghi Chú

1. **Part 1 Integration**: Part 3 dùng `gaussian_eliminate` từ Part 1 qua dynamic loading. Đảm bảo Part 1 đã được hoàn thiện trước khi chạy Part 3.

2. **Precision**: Tất cả tính toán dùng `float` (64-bit IEEE). Các so sánh với epsilon (EPS = 1e-12) để xử lý sai số số học.

3. **Matrix Storage**: Ma trận lưu dưới dạng list 2 chiều (row-major). Chỉ lưu phần cần thiết (ví dụ Cholesky chỉ lưu phần dưới L).

4. **Benchmark Note**: Nếu máy yếu hoặc muốn chạy nhanh, giảm `sizes` hay `repeats` trong `benchmark.py`.

5. **Stability Study**: Hilbert matrix có κ(A) rất lớn (10^5 ~ 10^13), dùng để minh chứng tác động của condition number.

---
