# -*- coding: utf-8 -*-
"""
问题3：确定最小螺距，使龙头前把手能够沿该螺线盘入到直径9m圆（半径R=4.5m）的边界，且在此之前不发生碰撞。

方法（与 q2 一致的链条与碰撞检测，按螺距搜索）：
- 复用“画圆找交点”从龙头向外生成整条把手链；速度用投影关系（仅用于一致性，可不参与判定）。
- 碰撞检测：将每块板建模为有方向矩形（中心线两端各外伸0.275m，半宽0.15m），用面积法判断顶点是否落入其它（非相邻）矩形。
- 搜索最小螺距：
  1) 从 p=0.55m 开始，以 0.01m 逐步减小；每个 p，计算“到达边界所需时间 t_boundary”，并从 t=0 到 t_boundary 逐秒推进，若期间出现碰撞则判定失败，否则通过。
  2) 找到首个失败的 p 和其上一个成功的 p 形成区间 [p_fail, p_pass] 后，使用二分法将区间收敛到 1e-4 m。

输出：在控制台打印最小可行螺距与对应边界时间；如需，可将过程写入 data/result3_pitch_search.xlsx（可选）。
"""
from __future__ import annotations
import math
from typing import List, Tuple
import pandas as pd

# -------------------- 固定几何参数（与题意一致） --------------------
V_HEAD = 1.0  # m/s 龙头弧长速度
N_BODY = 221
L_HEAD = 3.41 - 0.55  # 2.86 m
L_BODY = 2.20 - 0.55  # 1.65 m
N_FRONT_HANDLES = 1 + N_BODY + 1  # 223 前把手
THETA0_HEAD = 2.0 * math.pi * 16.0  # 初始在第16圈

BOARD_HALF_W = 0.30 / 2.0  # 0.15 m
END_EXT = 0.275  # m（孔到板端）

R_TURN = 9.0 / 2.0  # 掉头空间半径=4.5m

# 数值参数
DT_COARSE = 1.0  # s，时间推进步长
TIME_EPS = 1e-3  # s，二分终止阈值（用于更细致时可扩展）
PITCH_STEP = 0.01  # m，粗步长
PITCH_TOL = 1e-4  # m，二分终止


# -------------------- 螺线与弧长（以 pitch 为参数） --------------------
def B_from_pitch(pitch: float) -> float:
    return pitch / (2.0 * math.pi)


def spiral_point(theta: float, B: float) -> Tuple[float, float]:
    r = B * theta
    return r * math.cos(theta), r * math.sin(theta)


def spiral_tangent_unit(theta: float, B: float) -> Tuple[float, float]:
    cx, sx = math.cos(theta), math.sin(theta)
    tx = B * (cx - theta * sx)
    ty = B * (sx + theta * cx)
    norm = math.hypot(tx, ty)
    return (tx / norm, ty / norm)


def arc_len_primitive(theta: float, B: float) -> float:
    return 0.5 * B * (theta * math.sqrt(1.0 + theta * theta) + math.asinh(theta))


def arc_len_derivative(theta: float, B: float) -> float:
    return B * math.sqrt(1.0 + theta * theta)


def theta_from_arclen(S_target: float, theta_high: float, B: float) -> float:
    # 在 [0, theta_high] 解 S(θ)=S_target（盘入）
    S0, SHigh = 0.0, arc_len_primitive(theta_high, B)
    if S_target <= S0:
        return 0.0
    if S_target >= SHigh:
        return theta_high
    # 先线性估计，再二分
    t = S_target / SHigh
    lo, hi = 0.0, theta_high
    theta = max(0.0, min(theta_high, t * theta_high))
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if arc_len_primitive(mid, B) < S_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chord_distance(theta0: float, theta1: float, B: float) -> float:
    x0, y0 = spiral_point(theta0, B)
    x1, y1 = spiral_point(theta1, B)
    return math.hypot(x1 - x0, y1 - y0)


def solve_next_theta_by_chord(theta_curr: float, L: float, B: float) -> float:
    slope = arc_len_derivative(theta_curr, B)
    dtheta_lo = max(L / max(slope, 1e-16), 1e-9)
    lo = theta_curr + dtheta_lo
    hi = theta_curr + 1.5 * dtheta_lo
    for _ in range(80):
        if chord_distance(theta_curr, hi, B) >= L:
            break
        hi += 1.5 * dtheta_lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        d = chord_distance(theta_curr, mid, B)
        if abs(d - L) < 1e-10:
            return mid
        if d < L:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -------------------- 链条与速度 --------------------
def build_theta_chain(theta_head: float, B: float) -> List[float]:
    thetas: List[float] = [theta_head]
    thetas.append(solve_next_theta_by_chord(thetas[-1], L_HEAD, B))
    total_steps = N_FRONT_HANDLES - 1
    for _ in range(total_steps):
        thetas.append(solve_next_theta_by_chord(thetas[-1], L_BODY, B))
    thetas.append(solve_next_theta_by_chord(thetas[-1], L_BODY, B))
    return thetas


def compute_positions(thetas: List[float], B: float) -> List[Tuple[float, float]]:
    return [spiral_point(th, B) for th in thetas]


def compute_speeds(thetas: List[float], B: float) -> List[Tuple[float, float]]:
    n = len(thetas)
    pts = compute_positions(thetas, B)
    T = [spiral_tangent_unit(th, B) for th in thetas]
    U: List[Tuple[float, float]] = []
    for i in range(n - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        L = math.hypot(dx, dy)
        U.append((dx / L, dy / L))
    v_mag = [0.0] * n
    v_mag[0] = V_HEAD
    for i in range(n - 1):
        dot_i = T[i][0] * U[i][0] + T[i][1] * U[i][1]
        dot_ip1 = T[i + 1][0] * U[i][0] + T[i + 1][1] * U[i][1]
        dot_ip1 = math.copysign(max(abs(dot_ip1), 1e-12), dot_ip1)
        v_mag[i + 1] = abs(v_mag[i] * (dot_i / dot_ip1))
    v_vec = [(-v_mag[i] * T[i][0], -v_mag[i] * T[i][1]) for i in range(n)]
    return v_vec


# -------------------- 矩形碰撞检测（面积法） --------------------
def get_board_rectangles(points: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
    rects: List[List[Tuple[float, float]]] = []
    for i in range(N_FRONT_HANDLES):  # 223 块板
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        vx, vy = -uy, ux
        ax, ay = x0 - END_EXT * ux, y0 - END_EXT * uy
        bx, by = x1 + END_EXT * ux, y1 + END_EXT * uy
        p1 = (ax + BOARD_HALF_W * vx, ay + BOARD_HALF_W * vy)
        p2 = (bx + BOARD_HALF_W * vx, by + BOARD_HALF_W * vy)
        p3 = (bx - BOARD_HALF_W * vx, by - BOARD_HALF_W * vy)
        p4 = (ax - BOARD_HALF_W * vx, ay - BOARD_HALF_W * vy)
        rects.append([p1, p2, p3, p4])
    return rects


def triangle_area(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)


def point_in_rectangle(point: Tuple[float, float], rect: List[Tuple[float, float]]) -> bool:
    p1, p2, p3, p4 = rect
    rect_area = triangle_area(p1, p2, p3) + triangle_area(p1, p3, p4)
    a1 = triangle_area(point, p1, p2)
    a2 = triangle_area(point, p2, p3)
    a3 = triangle_area(point, p3, p4)
    a4 = triangle_area(point, p4, p1)
    return abs((a1+a2+a3+a4) - rect_area) < 1e-9


def has_collision(points: List[Tuple[float, float]]) -> bool:
    rects = get_board_rectangles(points)
    n = len(rects)
    for i in range(n):
        ri = rects[i]
        for vertex in ri:
            for j in range(n):
                if abs(j - i) <= 1:
                    continue
                if point_in_rectangle(vertex, rects[j]):
                    return True
    return False


# -------------------- 时间推进与判定 --------------------
def compute_state_at_time(pitch: float, t: float):
    B = B_from_pitch(pitch)
    S0 = arc_len_primitive(THETA0_HEAD, B)
    S_t = max(0.0, S0 - V_HEAD * t)
    theta_head = theta_from_arclen(S_t, THETA0_HEAD, B)
    thetas = build_theta_chain(theta_head, B)
    pts = compute_positions(thetas, B)
    return {"B": B, "thetas": thetas, "points": pts, "theta_head": theta_head}


def time_to_reach_boundary(pitch: float) -> float:
    B = B_from_pitch(pitch)
    r0 = B * THETA0_HEAD
    if r0 <= R_TURN:
        # 已在或内于边界，认为 t=0 即可到达（若需“由外盘入”，请确保 pitch > R_TURN/16）
        return 0.0
    theta_target = R_TURN / B
    S0 = arc_len_primitive(THETA0_HEAD, B)
    St = arc_len_primitive(theta_target, B)
    return max(0.0, (S0 - St) / V_HEAD)


def can_reach_without_collision(pitch: float, collect_path: bool = False):
    """判断在给定螺距下，龙头在到达边界之前是否会发生碰撞。
    返回 (ok, t_boundary, t_first_collision, trace_df)
    ok=True 表示无碰撞即可到达边界。
    若 collect_path=True，返回每秒的记录 DataFrame（可选）。
    """
    t_boundary = time_to_reach_boundary(pitch)
    # 粗推进：从 t=0 到 t_boundary
    rows = []
    t = 0.0
    while t < t_boundary - 1e-9:
        st = compute_state_at_time(pitch, t)
        if has_collision(st["points"]):
            if collect_path:
                df = pd.DataFrame(rows)
            else:
                df = None
            return False, t_boundary, t, df
        if collect_path:
            xh, yh = spiral_point(st["theta_head"], st["B"])
            rows.append({"t": t, "r_head": math.hypot(xh, yh)})
        t += DT_COARSE
    # 最后检查 t_boundary（精确到边界时刻）
    st = compute_state_at_time(pitch, t_boundary)
    if has_collision(st["points"]):
        if collect_path:
            df = pd.DataFrame(rows)
        else:
            df = None
        return False, t_boundary, t_boundary, df
    if collect_path:
        xh, yh = spiral_point(st["theta_head"], st["B"])
        rows.append({"t": t_boundary, "r_head": math.hypot(xh, yh)})
        df = pd.DataFrame(rows)
    else:
        df = None
    return True, t_boundary, None, df


# -------------------- 螺距搜索 --------------------
def search_min_pitch() -> Tuple[float, pd.DataFrame]:
    p_lo_fail = None
    p_hi_pass = None
    records = []

    # 起点 p=0.55m 向下每次 0.01m，直到失败或到下界（由外盘入需 p > R_TURN/16）
    p = 0.55
    p_min_outside = R_TURN / 16.0 + 1e-6
    while p > p_min_outside:
        ok, t_b, t_c, _ = can_reach_without_collision(p)
        records.append({"pitch": p, "ok": ok, "t_boundary": t_b, "t_collide": t_c})
        if not ok:
            p_lo_fail = p
            break
        p_hi_pass = p
        p = round(p - PITCH_STEP, 5)

    # 若起点就失败，需要向上扩大区间
    if p_hi_pass is None:
        p = 0.55 + PITCH_STEP
        while True:
            ok, t_b, t_c, _ = can_reach_without_collision(p)
            records.append({"pitch": p, "ok": ok, "t_boundary": t_b, "t_collide": t_c})
            if ok:
                p_hi_pass = p
                p_lo_fail = p - PITCH_STEP
                break
            p += PITCH_STEP

    # 若向下未出现失败（一直 ok），最小值受“由外盘入”约束
    if p_lo_fail is None:
        return p_min_outside, pd.DataFrame(records)

    # 二分搜索 [p_lo_fail, p_hi_pass]
    lo, hi = p_lo_fail, p_hi_pass
    for _ in range(40):
        if hi - lo <= PITCH_TOL:
            break
        mid = 0.5 * (lo + hi)
        ok, t_b, t_c, _ = can_reach_without_collision(mid)
        records.append({"pitch": mid, "ok": ok, "t_boundary": t_b, "t_collide": t_c})
        if ok:
            hi = mid
        else:
            lo = mid
    return hi, pd.DataFrame(records)


def main():
    p_min, df = search_min_pitch()
    print(f"最小可行螺距 p_min = {p_min:.6f} m")
    # 可选输出过程
    try:
        out_path = "d:/Files/Code/CUMCM2024/data/result3_pitch_search.xlsx"
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            df.sort_values("pitch").to_excel(writer, index=False, sheet_name="search")
        print(f"搜索过程已导出到: {out_path}")
    except Exception as e:
        print(f"导出过程表失败（可忽略）: {e}")


if __name__ == "__main__":
    main()
