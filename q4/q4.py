# -*- coding: utf-8 -*-
"""
Q4：螺距 1.7 m 的阿基米德螺线盘入，盘出与盘入中心对称；在直径 9 m 的调头区内，
用两段相切圆弧（半径 a 与 2a，且分别与盘入/盘出螺线相切）完成调头。以“开始调头（龙头
刚到达边界并开始进入圆弧）”为 t=0，输出 t∈[-100,100] s 全队位置与速度到 data/result4.xlsx；
并绘制关键时刻（-100,-50,0,50,100）整队连线图到 data/。

实现要点：
- 标准阿基米德螺线 r=Bθ，B=p/(2π)，引入全局旋转 PHI，使边界交点位于左下象限（≈225°）。
- s 参数统一路径：s<0 盘入螺线，s∈[0,L1] 为小圆弧，s∈[L1,L1+L2] 为大圆弧，s>Lturn 为盘出螺线。
- “画圆找交点”在统一路径上用弧长参数 s 做二分，天然覆盖“跨段”诸情况（螺线↔圆弧↔螺线）。
- 圆弧几何固定 θ*=θ_b=R/B（边界），按 a 搜索满足两圆外切且均在调头区内的最短弧长解。
- 日志：打印几何搜索与每秒头部所在段；可视化叠加螺线、两圆弧、调头区。
"""
from __future__ import annotations
import math
import os
import time
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

# -------------------- 常量与规格 --------------------
PITCH = 1.70                 # m，螺距
R_TURN = 9.0 / 2.0           # m，调头区半径
V_HEAD = 1.0                 # m/s，龙头弧长速度

N_BODY = 221
L_HEAD = 3.41 - 0.55         # 2.86 m（龙头前后孔距）
L_BODY = 2.20 - 0.55         # 1.65 m（其余孔距）
N_TOTAL_POINTS = 1 + N_BODY + 2   # 龙头 + 221节 + 尾前 + 尾后 = 224
LAST_INDEX = N_TOTAL_POINTS - 1

T_NEG, T_POS, DT = -100, 100, 1

# 数值容差
ANG_TOL = 1e-12
ITER_MAX = 80

# 全局旋转（由 main 设定使边界点落在左下象限）
PHI = 0.0


# -------------------- 低层工具 --------------------
def B_from_pitch(pitch: float) -> float:
	return pitch / (2.0 * math.pi)


def rotate_left(x: float, y: float) -> Tuple[float, float]:
	return -y, x


def norm2(x: float, y: float) -> float:
	return math.hypot(x, y)


# -------------------- 螺线与弧长 --------------------
def spiral_point(theta: float, B: float) -> Tuple[float, float]:
	r = B * theta
	ang = theta + PHI
	return r * math.cos(ang), r * math.sin(ang)


def spiral_tangent_unit(theta: float, B: float) -> Tuple[float, float]:
	ang = theta + PHI
	cx, sx = math.cos(ang), math.sin(ang)
	tx = B * (cx - theta * sx)
	ty = B * (sx + theta * cx)
	n = math.hypot(tx, ty)
	return tx / n, ty / n


def arc_len_primitive(theta: float, B: float) -> float:
	return 0.5 * B * (theta * math.sqrt(1.0 + theta * theta) + math.asinh(theta))


def theta_from_S(St: float, B: float, theta_lo: float = 0.0) -> float:
	# 用二分反解 S(θ)=St，θ∈[theta_lo, θ_hi]
	# 近似上界 θ≈sqrt(2S/B)
	theta_est = math.sqrt(max(0.0, 2.0 * St / max(B, 1e-16))) if St > 0 else 0.0
	lo = max(0.0, theta_lo)
	hi = max(lo + 1.0, theta_est + 5.0)
	for _ in range(ITER_MAX):
		mid = 0.5 * (lo + hi)
		if arc_len_primitive(mid, B) < St:
			lo = mid
		else:
			hi = mid
	return 0.5 * (lo + hi)


# -------------------- 两圆弧调头几何 --------------------
class TurnGeometry:
	def __init__(self, a: float, theta_b: float, C1: Tuple[float, float], C2: Tuple[float, float],
				 Pin: Tuple[float, float], Pout: Tuple[float, float], B: float,
				 sign_pair: Tuple[int, int]):
		self.a = a
		self.theta_b = theta_b
		self.C1 = C1  # 先经过的大圆弧的圆心
		self.C2 = C2  # 后经过的小圆弧的圆心
		self.Pin = Pin
		self.Pout = Pout
		self.B = B
		self.sign_pair = sign_pair
		# 半径：先走大圆弧(2a)，再走小圆弧(a)
		self.R1 = 2.0 * a
		self.R2 = 1.0 * a
		# 外切点：位于 C1->C2 方向，按 R1 : R2 = 2 : 1 分割，距离 C1 的比例为 R1/(R1+R2)=2/3
		t = self.R1 / (self.R1 + self.R2)
		self.M = (C1[0] + (C2[0] - C1[0]) * t, C1[1] + (C2[1] - C1[1]) * t)
		# 预计算弧长与方向
		self.L1 = None
		self.L2 = None
		self.arc1_ccw = True
		self.arc2_ccw = True
		self.arc1_start = 0.0
		self.arc2_start = 0.0
		self.arc1_dtheta = 0.0
		self.arc2_dtheta = 0.0
		self._precompute_arc_params()

	@staticmethod
	def _angle(cx: float, cy: float, px: float, py: float) -> float:
		return math.atan2(py - cy, px - cx)

	def _precompute_arc_params(self) -> None:
		# 圆1：C1，R=2a，从 Pin 到 M
		ang_s = self._angle(self.C1[0], self.C1[1], self.Pin[0], self.Pin[1])
		ang_e = self._angle(self.C1[0], self.C1[1], self.M[0], self.M[1])
		# 方向由与盘入切向一致确定
		Tin = spiral_tangent_unit(self.theta_b, self.B)
		rad = (self.Pin[0] - self.C1[0], self.Pin[1] - self.C1[1])
		t_ccw = rotate_left(rad[0], rad[1])
		# 归一到 R1
		t_ccw = (t_ccw[0] / self.R1, t_ccw[1] / self.R1)
		# 在边界处，沿 s 增大是进入大圆弧，需与 -Tin 对齐
		self.arc1_ccw = (t_ccw[0] * (-Tin[0]) + t_ccw[1] * (-Tin[1])) >= 0

		def ang_delta(a1: float, a2: float, ccw: bool) -> float:
			d = (a2 - a1) % (2 * math.pi)
			return d if ccw else (d - 2 * math.pi)

		d1 = ang_delta(ang_s, ang_e, self.arc1_ccw)
		self.arc1_start = ang_s
		self.arc1_dtheta = d1
		self.L1 = self.R1 * abs(d1)

		# 圆2：C2, R=a，从 M 到 Pout；方向需与圆1在 M 处切向一致
		ang2_s = self._angle(self.C2[0], self.C2[1], self.M[0], self.M[1])
		ang2_e = self._angle(self.C2[0], self.C2[1], self.Pout[0], self.Pout[1])
		# 圆1在 M 处的切向
		rad1m = (self.M[0] - self.C1[0], self.M[1] - self.C1[1])
		t1_ccw = rotate_left(rad1m[0], rad1m[1])
		t1_ccw = (t1_ccw[0] / self.R1, t1_ccw[1] / self.R1)
		if not self.arc1_ccw:
			t1_ccw = (-t1_ccw[0], -t1_ccw[1])
		# 圆2在 M 处的逆时针切向
		rad2m = (self.M[0] - self.C2[0], self.M[1] - self.C2[1])
		t2_ccw = rotate_left(rad2m[0], rad2m[1])
		t2_ccw = (t2_ccw[0] / self.R2, t2_ccw[1] / self.R2)
		self.arc2_ccw = (t2_ccw[0] * t1_ccw[0] + t2_ccw[1] * t1_ccw[1]) >= 0
		d2 = ang_delta(ang2_s, ang2_e, self.arc2_ccw)
		self.arc2_start = ang2_s
		self.arc2_dtheta = d2
		self.L2 = self.R2 * abs(d2)

	# 位姿 on arcs
	def pos_on_arc1(self, s_local: float) -> Tuple[float, float]:
		frac = max(0.0, min(1.0, s_local / max(self.L1, 1e-16)))
		ang = self.arc1_start + frac * self.arc1_dtheta
		return (self.C1[0] + self.R1 * math.cos(ang), self.C1[1] + self.R1 * math.sin(ang))

	def tan_on_arc1(self, s_local: float) -> Tuple[float, float]:
		frac = max(0.0, min(1.0, s_local / max(self.L1, 1e-16)))
		ang = self.arc1_start + frac * self.arc1_dtheta
		t = rotate_left(math.cos(ang), math.sin(ang))
		return t if self.arc1_ccw else (-t[0], -t[1])

	def pos_on_arc2(self, s_local: float) -> Tuple[float, float]:
		frac = max(0.0, min(1.0, s_local / max(self.L2, 1e-16)))
		ang = self.arc2_start + frac * self.arc2_dtheta
		return (self.C2[0] + self.R2 * math.cos(ang), self.C2[1] + self.R2 * math.sin(ang))

	def tan_on_arc2(self, s_local: float) -> Tuple[float, float]:
		frac = max(0.0, min(1.0, s_local / max(self.L2, 1e-16)))
		ang = self.arc2_start + frac * self.arc2_dtheta
		t = rotate_left(math.cos(ang), math.sin(ang))
		return t if self.arc2_ccw else (-t[0], -t[1])


def _arc_inside_turning_area(C: Tuple[float, float], R: float,
							 P0: Tuple[float, float], P1: Tuple[float, float], ccw: bool,
							 samples: int = 60) -> bool:
	# 在圆弧 P0->P1 上采样检查 |x|<=R_TURN
	a0 = math.atan2(P0[1] - C[1], P0[0] - C[0])
	a1 = math.atan2(P1[1] - C[1], P1[0] - C[0])
	d = (a1 - a0) % (2 * math.pi)
	d = d if ccw else (d - 2 * math.pi)
	for i in range(samples + 1):
		t = i / samples
		ang = a0 + d * t
		x = C[0] + R * math.cos(ang)
		y = C[1] + R * math.sin(ang)
		if norm2(x, y) > R_TURN + 1e-8:
			return False
	return True


def search_min_turn(theta_b: float, B: float) -> Tuple[TurnGeometry, float]:
	Pin = spiral_point(theta_b, B)
	Pout = (-Pin[0], -Pin[1])
	Tin = spiral_tangent_unit(theta_b, B)
	Nin = rotate_left(Tin[0], Tin[1])
	# Nout = -Nin（由对称关系）
	candidates: List[Tuple[TurnGeometry, float]] = []
	R = R_TURN
	dotRN = Pin[0] * Nin[0] + Pin[1] * Nin[1]
	for s1, s2 in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
		# 由新的定义（先 2a 再 a），推导得到 | -2Pin - (2*s1 + s2)a Nin | = 3a
		k = 2 * s1 + s2
		# 由 | -2Pin - k a Nin | = 3a 推得：(k^2-9)a^2 + 4k(dotRN)a + 4R^2 = 0
		A = (k * k - 9.0)
		Bq = 4.0 * k * dotRN
		Cq = 4.0 * (R * R)
		a_solutions: List[float] = []
		if abs(A) < 1e-12:
			# 退化为一次
			if abs(Bq) > 1e-12:
				a_lin = -Cq / Bq
				if a_lin > 1e-8:
					a_solutions.append(a_lin)
		else:
			disc = Bq * Bq - 4.0 * A * Cq
			if disc >= 0.0:
				sqrtD = math.sqrt(disc)
				a1 = (-Bq + sqrtD) / (2.0 * A)
				a2 = (-Bq - sqrtD) / (2.0 * A)
				for aa in (a1, a2):
					if aa > 1e-8:
						a_solutions.append(aa)
		for a in a_solutions:
			# 先经过大圆弧(2a)的圆心 C1；再经过小圆弧(a)的圆心 C2
			C1 = (Pin[0] + s1 * 2 * a * Nin[0], Pin[1] + s1 * 2 * a * Nin[1])
			C2 = (-Pin[0] - s2 * a * Nin[0], -Pin[1] - s2 * a * Nin[1])
			tg = TurnGeometry(a, theta_b, C1, C2, Pin, Pout, B, (s1, s2))
			# 只要求采样点都在调头区内
			ok1 = _arc_inside_turning_area(C1, 2 * a, tg.Pin, tg.M, tg.arc1_ccw)
			ok2 = _arc_inside_turning_area(C2, 1 * a, tg.M, tg.Pout, tg.arc2_ccw)
			if ok1 and ok2 and tg.L1 > 1e-8 and tg.L2 > 1e-8:
				candidates.append((tg, a))
				print(f"[候选] 符号({s1},{s2}) a={a:.6f} L1={tg.L1:.6f} L2={tg.L2:.6f}")
	if not candidates:
		raise RuntimeError("未找到可行的调头几何，请检查参数/旋转。")
	# 选最短弧长
	best, a_best = min(candidates, key=lambda it: it[0].L1 + it[0].L2)
	print(f"[结果] a*={a_best:.6f}, L_turn={best.L1 + best.L2:.6f}")
	return best, a_best


# -------------------- 统一路径 γ(s) --------------------
class CompositePath:
	def __init__(self, tg: TurnGeometry):
		self.tg = tg
		self.B = tg.B
		self.theta_b = tg.theta_b
		self.S_b = arc_len_primitive(self.theta_b, self.B)
		self.Lturn = tg.L1 + tg.L2

	def pos(self, s: float) -> Tuple[float, float]:
		if s < 0:  # 盘入（外侧）
			St = self.S_b - s
			theta = theta_from_S(St, self.B, theta_lo=self.theta_b)
			return spiral_point(theta, self.B)
		if s <= self.tg.L1:
			return self.tg.pos_on_arc1(s)
		if s <= self.Lturn:
			return self.tg.pos_on_arc2(s - self.tg.L1)
		# 盘出（外侧）
		St = self.S_b + (s - self.Lturn)
		theta = theta_from_S(St, self.B, theta_lo=self.theta_b)
		pin = spiral_point(theta, self.B)
		return (-pin[0], -pin[1])

	def tan(self, s: float) -> Tuple[float, float]:
		if s < 0:
			St = self.S_b - s
			theta = theta_from_S(St, self.B, theta_lo=self.theta_b)
			Tin = spiral_tangent_unit(theta, self.B)
			return -Tin[0], -Tin[1]
		if s <= self.tg.L1:
			return self.tg.tan_on_arc1(s)
		if s <= self.Lturn:
			return self.tg.tan_on_arc2(s - self.tg.L1)
		St = self.S_b + (s - self.Lturn)
		theta = theta_from_S(St, self.B, theta_lo=self.theta_b)
		Tin = spiral_tangent_unit(theta, self.B)
		Tout = (-Tin[0], -Tin[1])
		return Tout

	def locate_segment(self, s: float) -> str:
		if s < 0:
			return "in-spiral"
		if s <= self.tg.L1:
			return "arc1"
		if s <= self.Lturn:
			return "arc2"
		return "out-spiral"

	# 可视化底图：调头区、两螺线、两圆弧
	def draw_base(self, ax):
		ax.set_aspect('equal', adjustable='box')
		# 调头区
		circle = plt.Circle((0, 0), R_TURN, color='tab:blue', fill=False, linestyle='--', linewidth=1)
		shade = plt.Circle((0, 0), R_TURN, color='gold', alpha=0.15)
		ax.add_artist(shade)
		ax.add_artist(circle)
		# 盘入/盘出螺线（到边界外若干圈）
		thb = self.theta_b
		th_vals = [thb + i * 0.05 for i in range(0, 400)]  # 往外 20 弧度
		xin, yin = zip(*[spiral_point(th, self.B) for th in th_vals])
		xout, yout = zip(*[(-x, -y) for x, y in zip(xin, yin)])
		ax.plot(xin, yin, color='tab:red', lw=1, label='In-spiral')
		ax.plot(xout, yout, color='tab:blue', lw=1, label='Out-spiral')
		# 两圆弧
		# 先行大圆弧（R1=2a）
		ang_s1 = TurnGeometry._angle(self.tg.C1[0], self.tg.C1[1], self.tg.Pin[0], self.tg.Pin[1])
		ang_e1 = TurnGeometry._angle(self.tg.C1[0], self.tg.C1[1], self.tg.M[0], self.tg.M[1])
		angs1 = [ang_s1 + self.tg.arc1_dtheta * t for t in [i / 100.0 for i in range(101)]]
		x1 = [self.tg.C1[0] + self.tg.R1 * math.cos(a) for a in angs1]
		y1 = [self.tg.C1[1] + self.tg.R1 * math.sin(a) for a in angs1]
		ax.plot(x1, y1, color='k', lw=1.8, label='Arc1(2a)')
		# 后续小圆弧（R2=a）
		ang_s2 = TurnGeometry._angle(self.tg.C2[0], self.tg.C2[1], self.tg.M[0], self.tg.M[1])
		ang_e2 = TurnGeometry._angle(self.tg.C2[0], self.tg.C2[1], self.tg.Pout[0], self.tg.Pout[1])
		angs2 = [ang_s2 + self.tg.arc2_dtheta * t for t in [i / 100.0 for i in range(101)]]
		x2 = [self.tg.C2[0] + self.tg.R2 * math.cos(a) for a in angs2]
		y2 = [self.tg.C2[1] + self.tg.R2 * math.sin(a) for a in angs2]
		ax.plot(x2, y2, color='k', lw=1.8, linestyle='-')
		ax.grid(True, linestyle=':', linewidth=0.5)


# -------------------- 画圆找交点（在统一路径上） --------------------
def distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
	return math.hypot(q[0] - p[0], q[1] - p[1])


def solve_next_s_by_chord(path: CompositePath, s_curr: float, L: float) -> float:
	"""在路径上从 s_curr 向前（s 增大方向）寻找与当前点弦长为 L 的下一个 s。"""
	p0 = path.pos(s_curr)
	# 先指数增大上界，直至 chord>=L
	lo = s_curr + 1e-6
	hi = lo + max(L, 0.5)
	for _ in range(60):
		d = distance(p0, path.pos(hi))
		if d >= L:
			break
		hi += max(L, 0.5)
	for _ in range(ITER_MAX):
		mid = 0.5 * (lo + hi)
		d = distance(p0, path.pos(mid))
		if abs(d - L) < 1e-9:
			return mid
		if d < L:
			lo = mid
		else:
			hi = mid
	return 0.5 * (lo + hi)


def solve_prev_s_by_chord(path: CompositePath, s_curr: float, L: float) -> float:
	"""在路径上从 s_curr 向后（s 减小方向）寻找与当前点弦长为 L 的上一个 s。"""
	p0 = path.pos(s_curr)
	# 建立 [lo, hi] 区间，使得 d(lo)>=L, d(hi)<L，且 lo < hi < s_curr
	hi = s_curr - 1e-6
	lo = hi - max(L, 0.5)
	for _ in range(60):
		d = distance(p0, path.pos(lo))
		if d >= L:
			break
		lo -= max(L, 0.5)
	# 二分收缩区间
	for _ in range(ITER_MAX):
		mid = 0.5 * (lo + hi)
		d = distance(p0, path.pos(mid))
		if abs(d - L) < 1e-9:
			return mid
		if d >= L:
			lo = mid
		else:
			hi = mid
	return 0.5 * (lo + hi)


def build_chain_on_path(path: CompositePath, s_head: float) -> Tuple[List[float], List[Tuple[float, float]]]:
	"""从龙头开始，按“向后取”的方向依次得到龙身与龙尾的位置。
	约定：索引递增对应 s 递减（越靠后 s 越小），满足题意在 t=0 时龙身仍在盘入侧。
	"""
	s_list: List[float] = [s_head]
	# 先取与龙头相距 L_HEAD 的第一节（向后，s 变小）
	s_list.append(solve_prev_s_by_chord(path, s_list[-1], L_HEAD))
	# 再连续向后取 N_BODY 节 + 尾前（共 N_BODY+1 次），每次间距 L_BODY
	for _ in range(N_BODY + 1):  # 221 身体 + 尾前
		s_list.append(solve_prev_s_by_chord(path, s_list[-1], L_BODY))
	pts = [path.pos(s) for s in s_list]
	return s_list, pts


def compute_speeds_on_path(path: CompositePath, s_list: List[float]) -> Tuple[List[Tuple[float, float]], List[float]]:
	T = [path.tan(s) for s in s_list]
	U: List[Tuple[float, float]] = []
	for i in range(len(s_list) - 1):
		p = path.pos(s_list[i])
		q = path.pos(s_list[i + 1])
		dx, dy = q[0] - p[0], q[1] - p[1]
		L = math.hypot(dx, dy)
		U.append((dx / L, dy / L))
	v_mag = [0.0] * len(s_list)
	v_mag[0] = V_HEAD
	for i in range(len(U)):
		dot_i = T[i][0] * U[i][0] + T[i][1] * U[i][1]
		dot_ip1 = T[i + 1][0] * U[i][0] + T[i + 1][1] * U[i][1]
		dot_ip1 = math.copysign(max(abs(dot_ip1), 1e-12), dot_ip1)
		v_mag[i + 1] = abs(v_mag[i] * (dot_i / dot_ip1))
	v_vec = [(v_mag[i] * T[i][0], v_mag[i] * T[i][1]) for i in range(len(s_list))]
	return v_vec, v_mag


# -------------------- 导出与可视化 --------------------
def handle_names() -> List[str]:
	names = ["龙头"]
	for i in range(1, N_BODY + 1):
		names.append(f"第{i}节龙身")
	names.append("龙尾（前）")
	names.append("龙尾（后）")
	return names


def compute_one_time(path: CompositePath, t: int) -> Dict:
	s_head = V_HEAD * float(t)  # s=0 即边界点
	segment = path.locate_segment(s_head)
	print(f"[进度] t={t:>4d}s  段={segment}")
	s_list, pts = build_chain_on_path(path, s_head)
	v_vec, v_mag = compute_speeds_on_path(path, s_list)
	rows = []
	names = handle_names()
	for idx, name in enumerate(names):
		x, y = pts[idx]
		vx, vy = v_vec[idx]
		rows.append({
			"t": t,
			"index": idx,
			"name": name,
			"x": x,
			"y": y,
			"vx": vx,
			"vy": vy,
			"speed": v_mag[idx],
		})
	return {"records": rows}


def export_q4(df: pd.DataFrame, path_all: str, path_tables: str) -> None:
	df = df.sort_values(["t", "index"]).reset_index(drop=True)
	
	print("[导出] 开始生成横向表格格式...")
	
	# 获取所有时刻和节点
	all_times = sorted(df.t.unique())
	all_indices = sorted(df.index.unique())
	handle_names_list = handle_names()
	
	print(f"[导出] 处理 {len(all_indices)} 个节点，{len(all_times)} 个时刻")
	
	# 使用透视表方式快速生成横向数据
	df_pivot_x = df.pivot_table(index='index', columns='t', values='x', fill_value='')
	df_pivot_y = df.pivot_table(index='index', columns='t', values='y', fill_value='')
	df_pivot_speed = df.pivot_table(index='index', columns='t', values='speed', fill_value='')
	
	print("[导出] 生成位置表...")
	# 位置表：交替x,y行
	pos_data = []
	for idx in all_indices:
		name = handle_names_list[idx] if idx < len(handle_names_list) else f"节点{idx}"
		
		# x坐标行
		row_x = {"": f"{name}x (m)"}
		for t in all_times:
			val = df_pivot_x.loc[idx, t] if (idx in df_pivot_x.index and t in df_pivot_x.columns) else ''
			row_x[f"{t} s"] = round(val, 6) if val != '' else ''
		pos_data.append(row_x)
		
		# y坐标行
		row_y = {"": f"{name}y (m)"}
		for t in all_times:
			val = df_pivot_y.loc[idx, t] if (idx in df_pivot_y.index and t in df_pivot_y.columns) else ''
			row_y[f"{t} s"] = round(val, 6) if val != '' else ''
		pos_data.append(row_y)
	
	print("[导出] 生成速度表...")
	# 速度表
	spd_data = []
	for idx in all_indices:
		name = handle_names_list[idx] if idx < len(handle_names_list) else f"节点{idx}"
		
		row_spd = {"": f"{name} (m/s)"}
		for t in all_times:
			val = df_pivot_speed.loc[idx, t] if (idx in df_pivot_speed.index and t in df_pivot_speed.columns) else ''
			row_spd[f"{t} s"] = round(val, 6) if val != '' else ''
		spd_data.append(row_spd)
	
	print("[导出] 写入Excel文件...")
	# 安全保存函数
	def _safe_write(path: str, pos_data: List[Dict], spd_data: List[Dict]):
		try:
			with pd.ExcelWriter(path, engine="xlsxwriter") as w:
				print(f"[导出] 写入位置表 ({len(pos_data)} 行)...")
				pd.DataFrame(pos_data).to_excel(w, index=False, sheet_name="position")
				print(f"[导出] 写入速度表 ({len(spd_data)} 行)...")
				pd.DataFrame(spd_data).to_excel(w, index=False, sheet_name="speed")
			print(f"[导出] 成功写入 {path}")
			return path
		except PermissionError:
			base, ext = os.path.splitext(path)
			alt = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
			with pd.ExcelWriter(alt, engine="xlsxwriter") as w:
				pd.DataFrame(pos_data).to_excel(w, index=False, sheet_name="position")
				pd.DataFrame(spd_data).to_excel(w, index=False, sheet_name="speed")
			print(f"[提示] {os.path.basename(path)} 被占用，已另存为 {os.path.basename(alt)}")
			return alt
		except Exception as e:
			print(f"[错误] 写入Excel时出错: {e}")
			raise
	
	# 保存主表为横向格式
	_safe_write(path_all, pos_data, spd_data)

	
	# 为了兼容原有的采样表，这里保留简化版本
	try:
		with pd.ExcelWriter(path_tables, engine="xlsxwriter") as w:
			# 只保存前20节的采样数据作为tables
			focus_sample = [-100, -50, 0, 50, 100]
			picks_sample = [0] + list(range(1, 21)) + [LAST_INDEX]
			
			pos_sample_rows = []
			spd_sample_rows = []
			
			for idx in picks_sample:
				name = handle_names_list[idx] if idx < len(handle_names_list) else f"节点{idx}"
				
				# 位置行
				row_x = {"": f"{name}x (m)"}
				row_y = {"": f"{name}y (m)"}
				for tt in focus_sample:
					sub = df[(df.t == tt) & (df.index == idx)]
					if len(sub) > 0:
						row_x[f"{tt} s"] = round(sub.iloc[0].x, 6)
						row_y[f"{tt} s"] = round(sub.iloc[0].y, 6)
					else:
						row_x[f"{tt} s"] = ""
						row_y[f"{tt} s"] = ""
				pos_sample_rows.append(row_x)
				pos_sample_rows.append(row_y)
				
				# 速度行
				row_spd = {"": f"{name} (m/s)"}
				for tt in focus_sample:
					sub = df[(df.t == tt) & (df.index == idx)]
					if len(sub) > 0:
						row_spd[f"{tt} s"] = round(sub.iloc[0].speed, 6)
					else:
						row_spd[f"{tt} s"] = ""
				spd_sample_rows.append(row_spd)
			
			pd.DataFrame(pos_sample_rows).to_excel(w, index=False, sheet_name="position_sample")
			pd.DataFrame(spd_sample_rows).to_excel(w, index=False, sheet_name="speed_sample")
	except PermissionError:
		base, ext = os.path.splitext(path_tables)
		alt = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
		with pd.ExcelWriter(alt, engine="xlsxwriter") as w:
			pd.DataFrame(pos_sample_rows).to_excel(w, index=False, sheet_name="position_sample")
			pd.DataFrame(spd_sample_rows).to_excel(w, index=False, sheet_name="speed_sample")
		print(f"[提示] {os.path.basename(path_tables)} 被占用，已另存为 {os.path.basename(alt)}")


def visualize_times(path: CompositePath, times: List[int], out_dir: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	
	# 设置中文字体，避免字体警告
	try:
		plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
		plt.rcParams['axes.unicode_minus'] = False
	except:
		# 如果设置字体失败，使用英文标题
		pass
	
	# 合图
	fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4), constrained_layout=True)
	if len(times) == 1:
		axes = [axes]
	for ax, tt in zip(axes, times):
		path.draw_base(ax)
		s_head = V_HEAD * float(tt)
		_, pts = build_chain_on_path(path, s_head)
		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		ax.plot(xs, ys, 'k.-', ms=2)
		ax.plot(xs[0], ys[0], 'ro', ms=4)
		ax.set_title(f"t={tt}s")
	# 使用英文标题避免字体问题
	fig.suptitle("Q4 Key Time Chain Visualization", fontsize=12)
	fig.savefig(os.path.join(out_dir, "q4_key_times.png"), dpi=150)
	plt.close(fig)
	# 单张
	for tt in times:
		fig, ax = plt.subplots(figsize=(5, 5))
		path.draw_base(ax)
		s_head = V_HEAD * float(tt)
		_, pts = build_chain_on_path(path, s_head)
		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		ax.plot(xs, ys, 'k.-', ms=2)
		ax.plot(xs[0], ys[0], 'ro', ms=4)
		ax.set_title(f"t={tt}s")
		fig.savefig(os.path.join(out_dir, f"q4_t{tt}.png"), dpi=150)
		plt.close(fig)


# -------------------- 主流程 --------------------
def main():
	B = B_from_pitch(PITCH)
	# 设旋转，使边界点在左下象限（≈225°）
	global PHI
	theta_b = R_TURN / B
	ang0 = theta_b % (2 * math.pi)
	target = 225.0 / 180.0 * math.pi
	PHI = target - ang0
	print(f"[初始化] B={B:.6f}, θ_b={theta_b:.6f}, PHI={PHI:.6f}")

	# 搜索最短可行调头弧（固定 θ_b）
	tg, a_best = search_min_turn(theta_b, B)
	path = CompositePath(tg)
	print(f"[几何] a={a_best:.6f}, L1={tg.L1:.6f}, L2={tg.L2:.6f}, Lturn={path.Lturn:.6f}")

	# 逐秒生成数据并导出
	all_rows: List[Dict] = []
	for t in range(T_NEG, T_POS + 1, DT):
		out = compute_one_time(path, t)
		all_rows.extend(out["records"])
	df = pd.DataFrame(all_rows)
	visualize_times(path, [-100, -50, 0, 50, 100], out_dir="d:/Files/Code/CUMCM2024/data")
	export_q4(df,
			  path_all="d:/Files/Code/CUMCM2024/data/result4.xlsx",
			  path_tables="d:/Files/Code/CUMCM2024/data/result_q4_tables.xlsx")
	print("导出完成：data/result4.xlsx, data/result_q4_tables.xlsx 及关键时刻图片")


if __name__ == "__main__":
	main()

