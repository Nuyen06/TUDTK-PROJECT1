from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
from pathlib import Path
import sys
import importlib

# =========================
# KHAI BÁO KIỂU DỮ LIỆU CHUNG
# =========================
# Matrix: ma trận số thực biểu diễn bằng list 2 chiều
# Vector: vector số thực biểu diễn bằng list 1 chiều
Matrix = List[List[float]]
Vector = List[float]
EPS = 1e-12


def _load_part1_gaussian_eliminate():
	# Nạp hàm Gaussian của Part 1 theo đường dẫn động để tái sử dụng code cũ.
	# Mục tiêu: bảo đảm Part 3 dùng đúng phần cài đặt đã hoàn thiện ở Part 1.
	part3_dir = Path(__file__).resolve().parent
	project_root = part3_dir.parent
	part1_dir = project_root / "part1"
	part1_dir_str = str(part1_dir)
	if part1_dir_str not in sys.path:
		sys.path.insert(0, part1_dir_str)
	gaussian_module = importlib.import_module("gaussian")
	return gaussian_module.gaussian_eliminate


def copy_matrix(A: Matrix) -> Matrix:
	# Sao chép sâu ma trận để tránh sửa trực tiếp dữ liệu đầu vào.
	return [row[:] for row in A]


def validate_system(A: Matrix, b: Vector) -> None:
	# B1: kiểm tra dữ liệu đầu vào cơ bản.
	if not A or not A[0]:
		raise ValueError("Ma trận A không được rỗng")
	n = len(A)
	m = len(A[0])
	# B2: hệ phải vuông để giải trực tiếp Ax=b.
	if n != m:
		raise ValueError("Ma trận A phải là ma trận vuông")
	for row in A:
		if len(row) != m:
			raise ValueError("Các hàng của A không cùng số cột")
	# B3: kích thước b phải khớp số dòng của A.
	if len(b) != n:
		raise ValueError("Kích thước của b phải bằng số hàng của A")


def is_symmetric(A: Matrix, tol: float = 1e-10) -> bool:
	# Kiểm tra đối xứng theo nửa trên ma trận để giảm phép so sánh dư thừa.
	n = len(A)
	for i in range(n):
		for j in range(i + 1, n):
			if abs(A[i][j] - A[j][i]) > tol:
				return False
	return True


def norm2(v: Vector) -> float:
	# Chuẩn Euclid (chuẩn 2): sqrt(sum(v_i^2)).
	return math.sqrt(sum(x * x for x in v))


def mat_vec_mul(A: Matrix, x: Vector) -> Vector:
	# Nhân ma trận - vector: (Ax)_i = sum_j A_ij * x_j.
	return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]


def relative_residual(A: Matrix, x: Vector, b: Vector, eps: float = EPS) -> float:
	# Residual tương đối: ||Ax-b|| / ||b||.
	Ax = mat_vec_mul(A, x)
	r = [ai - bi for ai, bi in zip(Ax, b)]
	denom = norm2(b)
	if denom < eps:
		return norm2(r)
	return norm2(r) / denom


def forward_substitution(L: Matrix, b: Vector, eps: float = EPS) -> Vector:
	# Giải Ly=b theo chiều từ trên xuống.
	# Vì mỗi y[i] phụ thuộc các y[0..i-1], ta xử lý i tăng dần.
	n = len(L)
	y = [0.0] * n
	for i in range(n):
		# s = tổng các thành phần đã biết ở bên trái đường chéo.
		s = sum(L[i][j] * y[j] for j in range(i))
		diag = L[i][i]
		if abs(diag) < eps:
			raise ValueError("Pivot bằng 0 khi thế tiến")
		y[i] = (b[i] - s) / diag
	return y


def back_substitution(U: Matrix, b: Vector, eps: float = EPS) -> Vector:
	# Giải Ux=b theo chiều từ dưới lên.
	# Vì mỗi x[i] phụ thuộc các x[i+1..n-1], ta xử lý i giảm dần.
	n = len(U)
	x = [0.0] * n
	for i in range(n - 1, -1, -1):
		# s = tổng các thành phần bên phải đường chéo đã biết.
		s = sum(U[i][j] * x[j] for j in range(i + 1, n))
		diag = U[i][i]
		if abs(diag) < eps:
			raise ValueError("Pivot bằng 0 khi thế lùi")
		x[i] = (b[i] - s) / diag
	return x


def gaussian_solve_pp(A: Matrix, b: Vector, eps: float = EPS) -> Tuple[Vector, Dict]:
	# B1: kiểm tra đầu vào và tạo ma trận tăng cường [A|b].
	validate_system(A, b)
	n = len(A)
	M = [A[i][:] + [b[i]] for i in range(n)]
	swaps = 0

	# B2: khử Gauss có chọn pivot cục bộ theo từng cột.
	for k in range(n):
		# Chọn hàng có trị tuyệt đối lớn nhất tại cột k để giảm sai số số học.
		pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
		if abs(M[pivot_row][k]) < eps:
			raise ValueError("Ma trận suy biến hoặc gần suy biến")
		if pivot_row != k:
			# Hoán đổi hàng để đưa pivot tốt nhất lên vị trí đang xét.
			M[k], M[pivot_row] = M[pivot_row], M[k]
			swaps += 1

		for i in range(k + 1, n):
			# Tính hệ số khử để triệt tiêu phần tử M[i][k].
			factor = M[i][k] / M[k][k]
			M[i][k] = 0.0
			for j in range(k + 1, n + 1):
				# Biến đổi hàng: dòng i = dòng i - factor * dòng k.
				M[i][j] -= factor * M[k][j]

	# B3: tách U và c, sau đó thế ngược để lấy nghiệm.
	U = [row[:n] for row in M]
	c = [row[n] for row in M]
	x = back_substitution(U, c, eps=eps)
	return x, {"swaps": swaps}


def gaussian_solve_part1(A: Matrix, b: Vector) -> Tuple[Vector, Dict]:
	"""Gọi trực tiếp solver Gauss đã cài ở Part 1 và quy đổi nghiệm về float."""
	# B1: xác thực hệ và gọi lại hàm Gaussian của Part 1.
	validate_system(A, b)
	gaussian_eliminate = _load_part1_gaussian_eliminate()
	# Part 1 trả về: (ma trận sau khử, nghiệm, số lần đổi hàng).
	_, x_raw, swap_count = gaussian_eliminate(A, b)

	# B2: chuẩn hóa các trường hợp nghiệm để dùng ổn định ở Part 3.
	if x_raw is None:
		raise ValueError("Hệ vô nghiệm theo kết quả Part 1")
	if isinstance(x_raw, list) and x_raw and isinstance(x_raw[0], str):
		raise ValueError("Hệ vô số nghiệm theo kết quả Part 1")

	# B3: đổi Fraction sang float để benchmark và tính residual.
	x = [float(v) for v in x_raw]
	return x, {"swaps": int(swap_count), "source": "part1.gaussian_eliminate"}


def cholesky_decomposition(A: Matrix, eps: float = EPS) -> Matrix:
	# B1: kiểm tra điều kiện Cholesky (vuông, đối xứng, xác định dương).
	n = len(A)
	if any(len(row) != n for row in A):
		raise ValueError("Ma trận A phải là ma trận vuông")
	if not is_symmetric(A):
		raise ValueError("Cholesky yêu cầu ma trận đối xứng")

	# B2: tính dần từng cột của L.
	L = [[0.0] * n for _ in range(n)]
	for j in range(n):
		# Tính phần tử đường chéo L[j][j].
		s = sum(L[j][k] ** 2 for k in range(j))
		diag_val = A[j][j] - s
		if diag_val <= eps:
			raise ValueError("Ma trận không xác định dương")
		L[j][j] = math.sqrt(diag_val)

		for i in range(j + 1, n):
			# Tính phần tử dưới đường chéo L[i][j].
			s = sum(L[i][k] * L[j][k] for k in range(j))
			L[i][j] = (A[i][j] - s) / L[j][j]

	return L


def solve_cholesky(A: Matrix, b: Vector, eps: float = EPS) -> Tuple[Vector, Dict]:
	# B1: phân rã A = L L^T.
	validate_system(A, b)
	L = cholesky_decomposition(A, eps=eps)
	# B2: giải Ly=b.
	y = forward_substitution(L, b, eps=eps)

	# B3: giải L^T x = y bằng thế ngược.
	# Lưu ý: L^T không tạo tường minh, truy cập qua L[j][i].
	n = len(A)
	x = [0.0] * n
	for i in range(n - 1, -1, -1):
		s = sum(L[j][i] * x[j] for j in range(i + 1, n))
		x[i] = (y[i] - s) / L[i][i]
	return x, {"factorization": "cholesky"}


def is_strictly_row_diagonally_dominant(A: Matrix) -> bool:
	# Chéo trội hàng nghiêm ngặt: |a_ii| > sum_{j!=i}|a_ij| với mọi i.
	n = len(A)
	for i in range(n):
		off = sum(abs(A[i][j]) for j in range(n) if j != i)
		if abs(A[i][i]) <= off:
			return False
	return True


def gauss_seidel(
	A: Matrix,
	b: Vector,
	x0: Optional[Vector] = None,
	tol: float = 1e-8,
	max_iter: int = 10000,
	eps: float = EPS,
) -> Tuple[Vector, Dict]:
	# B1: khởi tạo nghiệm ban đầu.
	validate_system(A, b)
	n = len(A)
	x = [0.0] * n if x0 is None else x0[:]

	# Chỉ báo điều kiện hội tụ thường gặp (không dùng để chặn thuật toán).
	diag_dom = is_strictly_row_diagonally_dominant(A)

	# B2: lặp cập nhật từng phần tử theo công thức Gauss-Seidel.
	for k in range(1, max_iter + 1):
		x_old = x[:]
		for i in range(n):
			diag = A[i][i]
			if abs(diag) < eps:
				raise ValueError("Gặp phần tử đường chéo bằng 0 trong Gauss-Seidel")
			# s1 dùng nghiệm mới x[0..i-1], s2 dùng nghiệm cũ x_old[i+1..].
			s1 = sum(A[i][j] * x[j] for j in range(i))
			s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
			x[i] = (b[i] - s1 - s2) / diag

		# B3: dừng khi sai khác giữa 2 lần lặp nhỏ hơn ngưỡng tol.
		diff_inf = max(abs(x[i] - x_old[i]) for i in range(n))
		if diff_inf < tol:
			return x, {"iterations": k, "converged": True, "diag_dominant": diag_dom}

	return x, {"iterations": max_iter, "converged": False, "diag_dominant": diag_dom}


if __name__ == "__main__":
	# Test nhanh dùng đúng bộ dữ liệu SPD đã thống nhất ở Part 2.
	A = [
		[4.0, 4.0, 2.0],
		[4.0, 8.0, 6.0],
		[2.0, 6.0, 9.0],
	]
	b = [14.0, 26.0, 23.0]

	# Chạy lần lượt 3 phương pháp để đối chiếu nghiệm và residual.
	x_gauss, info_gauss = gaussian_solve_part1(A, b)
	x_chol, info_chol = solve_cholesky(A, b)
	x_gs, info_gs = gauss_seidel(A, b, tol=1e-10)

	print("Gauss:", x_gauss, info_gauss, "| Sai số residual =", relative_residual(A, x_gauss, b))
	print("Cholesky:", x_chol, info_chol, "| Sai số residual =", relative_residual(A, x_chol, b))
	print("Gauss-Seidel:", x_gs, info_gs, "| Sai số residual =", relative_residual(A, x_gs, b))
