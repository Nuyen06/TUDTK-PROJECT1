from __future__ import annotations

from typing import Dict, List, Tuple
import random
import time
import math

import matplotlib.pyplot as plt

from solvers import gauss_seidel, gaussian_solve_part1, relative_residual, solve_cholesky

# =========================
# KHAI BÁO KIỂU DỮ LIỆU CHUNG
# =========================
Matrix = List[List[float]]
Vector = List[float]


def make_spd_matrix(n: int, seed: int) -> Matrix:
	# Tạo ma trận SPD bằng công thức A = R^T R + nI.
	# R là ma trận ngẫu nhiên, R^T R luôn bán xác định dương,
	# cộng thêm nI giúp A xác định dương rõ ràng và ổn định hơn.
	rng = random.Random(seed)
	R = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
	A = [[0.0] * n for _ in range(n)]

	# A = R^T R + nI is symmetric positive definite.
	for i in range(n):
		for j in range(n):
			# A[i][j] = tổng theo k của R[k][i] * R[k][j] (chính là R^T R).
			A[i][j] = sum(R[k][i] * R[k][j] for k in range(n))
		# Tăng phần tử đường chéo để cải thiện điều kiện số.
		A[i][i] += n
	return A


def make_random_vector(n: int, seed: int) -> Vector:
	# Sinh b ngẫu nhiên trong [-1, 1] để tránh thiên lệch đơn điệu.
	rng = random.Random(seed)
	return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def mean(values: List[float]) -> float:
	# Hàm trung bình đơn giản, trả nan nếu danh sách rỗng.
	return sum(values) / len(values) if values else float("nan")


def time_one_run(method_name: str, A: Matrix, b: Vector) -> Tuple[float, float, Dict]:
	# Đo thời gian 1 lần chạy và trả về residual tương ứng.
	t0 = time.perf_counter()

	# Chọn solver theo tên phương pháp.
	if method_name == "gauss_part1":
		x, info = gaussian_solve_part1(A, b)
	elif method_name == "cholesky":
		x, info = solve_cholesky(A, b)
	elif method_name == "gauss_seidel":
		x, info = gauss_seidel(A, b, tol=1e-8, max_iter=20000)
	else:
		raise ValueError(f"Phương pháp không hợp lệ: {method_name}")

	elapsed = time.perf_counter() - t0
	# Đánh giá chất lượng nghiệm bằng residual tương đối.
	rel_res = relative_residual(A, x, b)
	return elapsed, rel_res, info


def benchmark_sizes(
	sizes: List[int],
	repeats: int = 5,
	base_seed: int = 123,
) -> Dict[str, Dict[int, Dict[str, float]]]:
	# B1: cố định bộ phương pháp cần so sánh.
	# gauss_part1: Gauss kế thừa Part 1
	# cholesky   : direct method cho SPD
	# gauss_seidel: iterative method
	methods = ["gauss_part1", "cholesky", "gauss_seidel"]
	results: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in methods}

	# B2: duyệt từng kích thước ma trận.
	for n in sizes:
		# Dùng cùng A,b cho mọi phương pháp ở cùng n để so sánh công bằng.
		A = make_spd_matrix(n, seed=base_seed + n)
		b = make_random_vector(n, seed=base_seed + 2 * n)

		# B3: với mỗi phương pháp, chạy lặp nhiều lần rồi lấy trung bình.
		for m in methods:
			# Lưu thống kê theo nhiều lần chạy.
			times: List[float] = []
			residuals: List[float] = []
			converged_count = 0
			iter_counts: List[float] = []

			for _ in range(repeats):
				# Mỗi lần chạy thu cả thời gian lẫn residual.
				elapsed, rel_res, info = time_one_run(m, A, b)
				times.append(elapsed)
				residuals.append(rel_res)
				if m == "gauss_seidel":
					# Phương pháp lặp cần theo dõi hội tụ và số vòng lặp.
					if info.get("converged", False):
						converged_count += 1
					iter_counts.append(float(info.get("iterations", 0)))

			results[m][n] = {
				# time_avg và residual_avg là chỉ số chính dùng trong báo cáo.
				"time_avg": mean(times),
				"residual_avg": mean(residuals),
				"converged_ratio": converged_count / repeats if m == "gauss_seidel" else 1.0,
				"iter_avg": mean(iter_counts) if m == "gauss_seidel" else 0.0,
			}

			print(
				f"[n={n:4d}] {m:12s} | "
				f"thoi_gian_tb={results[m][n]['time_avg']:.6f}s | "
				f"residual_tb={results[m][n]['residual_avg']:.3e}"
			)

	return results


def plot_runtime_loglog(results: Dict[str, Dict[int, Dict[str, float]]], sizes: List[int]) -> None:
	# Vẽ đường thời gian của từng phương pháp trên thang log-log.
	plt.figure(figsize=(9, 6))

	for m in results:
		y = [results[m][n]["time_avg"] for n in sizes]
		plt.loglog(sizes, y, marker="o", label=m)

	n0 = sizes[0]
	ref0 = results["gauss_part1"][n0]["time_avg"]
	# Dựng đường chuẩn O(n^3) để quan sát xu hướng tăng độ phức tạp.
	# ref được scale theo mốc đầu tiên để dễ nhìn trên cùng hệ trục.
	ref = [ref0 * (n / n0) ** 3 for n in sizes]
	plt.loglog(sizes, ref, "--", label="Đường tham chiếu O(n^3)")

	plt.xlabel("Kích thước ma trận n")
	plt.ylabel("Thời gian chạy trung bình (giây)")
	plt.title("So sánh thời gian chạy trên thang log-log")
	plt.grid(True, which="both", linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.show()


def print_result_table(results: Dict[str, Dict[int, Dict[str, float]]], sizes: List[int]) -> None:
	# In bảng tổng hợp để đưa thẳng vào báo cáo.
	methods = list(results.keys())
	print("\n===== BẢNG TỔNG HỢP =====")
	for n in sizes:
		print(f"n = {n}")
		for m in methods:
			row = results[m][n]
			print(
				f"  {m:12s} | thoi_gian_tb={row['time_avg']:.6f}s | "
				f"residual_tb={row['residual_avg']:.3e} | "
				f"ty_le_hoi_tu={row['converged_ratio']:.2f} | "
				f"so_vong_tb={row['iter_avg']:.1f}"
			)


def condition_number_2(A: Matrix) -> float:
	"""Dùng NumPy để ước lượng điều kiện số chuẩn 2 phục vụ phân tích báo cáo."""
	# Chỉ dùng NumPy cho bước đánh giá, không dùng để cài solver.
	try:
		import numpy as np
		A_np = np.array(A, dtype=float)
		return float(np.linalg.cond(A_np, 2))
	except Exception:
		return float("nan")


def hilbert_matrix(n: int) -> Matrix:
	# Hilbert là ví dụ kinh điển của ma trận điều kiện kém khi n tăng.
	return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def make_spd_matrix_fixed(n: int, seed: int = 0) -> Matrix:
	# Wrapper tên rõ nghĩa để dùng trong case study ổn định số.
	return make_spd_matrix(n=n, seed=seed)


def stability_case_study(ns: List[int], seed: int = 0) -> Dict[str, List[Dict[str, float]]]:
	"""So sánh Hilbert và SPD theo condition number và residual của các solver."""
	# report chứa 2 nhánh dữ liệu để đối chiếu trực tiếp trong báo cáo.
	report: Dict[str, List[Dict[str, float]]] = {"hilbert": [], "spd": []}

	# B1: lặp qua từng kích thước cần phân tích.
	for n in ns:
		# Dùng b = vector 1 để giữ bài toán đơn giản, dễ tái lập.
		b = [1.0] * n

		# B2: nhánh Hilbert (thường điều kiện kém).
		H = hilbert_matrix(n)
		h_row: Dict[str, float] = {
			"n": float(n),
			"cond2": condition_number_2(H),
		}
		for method in ["gauss_part1", "cholesky", "gauss_seidel"]:
			try:
				_, res, info = time_one_run(method, H, b)
				h_row[f"res_{method}"] = res
				if method == "gauss_seidel":
					h_row["gs_converged"] = 1.0 if info.get("converged", False) else 0.0
			except Exception:
				# Nếu solver không áp dụng được, lưu nan để báo cáo minh bạch.
				h_row[f"res_{method}"] = float("nan")
				if method == "gauss_seidel":
					h_row["gs_converged"] = 0.0
		report["hilbert"].append(h_row)

		# B3: nhánh SPD ngẫu nhiên (thường điều kiện tốt hơn).
		S = make_spd_matrix_fixed(n, seed=seed + n)
		s_row: Dict[str, float] = {
			"n": float(n),
			"cond2": condition_number_2(S),
		}
		for method in ["gauss_part1", "cholesky", "gauss_seidel"]:
			try:
				_, res, info = time_one_run(method, S, b)
				s_row[f"res_{method}"] = res
				if method == "gauss_seidel":
					s_row["gs_converged"] = 1.0 if info.get("converged", False) else 0.0
			except Exception:
				s_row[f"res_{method}"] = float("nan")
				if method == "gauss_seidel":
					s_row["gs_converged"] = 0.0
		report["spd"].append(s_row)

	return report


def print_stability_report(report: Dict[str, List[Dict[str, float]]]) -> None:
	# In báo cáo ngắn gọn để mô tả mối liên hệ cond(A) và sai số thực nghiệm.
	print("\n===== ỔN ĐỊNH SỐ: HILBERT VS SPD =====")
	for name in ["hilbert", "spd"]:
		print(f"\n[{name.upper()}]")
		for row in report[name]:
			# fmt: chuẩn hóa cách hiển thị số, xử lý cả nan/inf.
			def fmt(v: float) -> str:
				if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
					return "nan"
				return f"{v:.3e}"

			print(
				f"n={int(row['n']):2d} | "
				f"cond2={fmt(row['cond2'])} | "
				f"res_gauss={fmt(row.get('res_gauss_part1', float('nan')))} | "
				f"res_cholesky={fmt(row.get('res_cholesky', float('nan')))} | "
				f"res_gs={fmt(row.get('res_gauss_seidel', float('nan')))} | "
				f"gs_hoi_tu={int(row.get('gs_converged', 0.0))}"
			)


def main() -> None:
	# Quy trình chính: benchmark runtime rồi chạy thêm case study ổn định số.
	# Có thể giảm sizes/repeats nếu máy yếu hoặc cần chạy nhanh.
	sizes = [50, 100, 200, 500, 1000]
	results = benchmark_sizes(sizes=sizes, repeats=5)
	print_result_table(results, sizes)
	plot_runtime_loglog(results, sizes)

	report = stability_case_study(ns=[5, 8, 10], seed=42)
	print_stability_report(report)


if __name__ == "__main__":
	main()
