# -*- coding: utf-8 -*-
"""
问题1计算脚本（按“思路c”实现）
- 等距螺线：r = b*theta，b = pitch/(2*pi)
- 龙头沿曲线弧长恒速 v_head = 1 m/s 顺时针盘入（theta 随时间减小）
- 各把手中心均在螺线上：从龙头前把手出发，逐段通过“圆与螺线交点（欧氏距离=同板两孔中心距）”向外寻找：
  [龙头前把手] -> [第1节龙身前把手] -> ... -> [第221节龙身前把手] -> [龙尾前把手] -> [龙尾后把手]
- 速度：用杆件长度约束的投影关系逐节传递（参考附件图示公式）
  对连接第 i 个“前把手”和第 i+1 个（同一板另一孔/下一节前把手）的板：
    令 U_i 为该板方向的单位向量（从 i 指向 i+1），T_i 为螺线切向单位向量（theta 增大方向）
    则（沿板方向的速度分量相等）: v_i * (T_i·U_i) = v_{i+1} * (T_{i+1}·U_i)
    注意盘入方向速度向量为 -v_i*T_i
- 输出：
  1) data/result1.xlsx  全时刻（t=0..300s，每秒）所有把手位置(x,y)、速度(vx,vy、speed)
  2) data/result_q1_tables.xlsx  论文表1/表2所需采样时刻与对象
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd

# -------------------- 常量与参数 --------------------
PITCH = 0.55  # m，题设螺距
B = PITCH / (2.0 * math.pi)  # r = B * theta
V_HEAD = 1.0  # m/s 龙头前把手弧长速度

# 板条数量：1 龙头 + 221 龙身 + 1 龙尾 = 223 节
N_BODY = 221

# 各板两孔中心距离（同一板上）
L_HEAD = 3.41 - 0.55  # m = 2.86
L_BODY = 2.20 - 0.55  # m = 1.65（龙身&龙尾）

# 链中“前把手”的数量（包含龙头与龙尾的前把手）
N_FRONT_HANDLES = 1 + N_BODY + 1  # 223
# 再加上龙尾后把手，共 N_FRONT_HANDLES + 1 个点

# 初始位置：第16圈 A 点，取 A 为极轴正向（theta = 2*pi*16）
THETA0_HEAD = 2.0 * math.pi * 16.0

# 时间设置
T_START = 0
T_END = 300
DT = 1

# 数值参数
ANG_TOL = 1e-12
DIST_TOL = 1e-10
NEWTON_MAX_IT = 50
BISECT_MAX_IT = 80


# -------------------- 几何与微分工具 --------------------

def spiral_point(theta: float) -> Tuple[float, float]:
    """等距螺线点：x = r*cos, y = r*sin；r = B*theta"""
    r = B * theta
    return r * math.cos(theta), r * math.sin(theta)


def spiral_tangent_unit(theta: float) -> Tuple[float, float]:
    """螺线按 theta 增大方向的切向单位向量 T_hat(theta)."""
    # P'(theta) = (dr/dθ cosθ - r sinθ, dr/dθ sinθ + r cosθ)
    #           = B * (cosθ - θ sinθ, sinθ + θ cosθ)
    cx, sx = math.cos(theta), math.sin(theta)
    tx = B * (cx - theta * sx)
    ty = B * (sx + theta * cx)
    norm = math.hypot(tx, ty)
    return (tx / norm, ty / norm)


def arc_len_primitive(theta: float) -> float:
    """S(θ) = ∫0^θ |P'(t)| dt = B * ∫0^θ sqrt(1 + t^2) dt
    = (B/2) * (θ*sqrt(1+θ^2) + asinh(θ))
    """
    return 0.5 * B * (theta * math.sqrt(1.0 + theta * theta) + math.asinh(theta))


def arc_len_derivative(theta: float) -> float:
    """dS/dθ = |P'(θ)| = B*sqrt(1+θ^2)"""
    return B * math.sqrt(1.0 + theta * theta)


def theta_from_arclen(target_S: float, theta_high: float) -> float:
    """已知 S(θ)=target_S，且 0<=θ<=theta_high（盘入），求 θ。
    用牛顿并辅以区间 [0, theta_high] 夹持。
    """
    # 边界
    S0, SHigh = 0.0, arc_len_primitive(theta_high)
    if target_S <= S0:
        return 0.0
    if target_S >= SHigh:
        return theta_high

    # 初值：线性插值 + 安全修正
    t = target_S / SHigh
    theta = max(0.0, min(theta_high, t * theta_high))

    for _ in range(NEWTON_MAX_IT):
        f = arc_len_primitive(theta) - target_S
        if abs(f) < DIST_TOL:
            return theta
        df = arc_len_derivative(theta)
        step = f / max(df, 1e-16)
        theta_new = theta - step
        if theta_new < 0.0 or theta_new > theta_high or abs(step) > 0.5 * (theta_high + 1.0):
            break
        theta = theta_new
    # 失败则二分
    lo, hi = 0.0, theta_high
    for _ in range(BISECT_MAX_IT):
        mid = 0.5 * (lo + hi)
        if arc_len_primitive(mid) < target_S:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chord_distance(theta0: float, theta1: float) -> float:
    x0, y0 = spiral_point(theta0)
    x1, y1 = spiral_point(theta1)
    return math.hypot(x1 - x0, y1 - y0)


def solve_next_theta_by_chord(theta_curr: float, L: float) -> float:
    """给定当前把手参数 theta_curr，求沿 theta 增大方向的下一个把手参数 theta_next，
    使得欧氏距离 |P(theta_next)-P(theta_curr)| = L。
    先用弧长近似给下界，再扩展上界，后用二分求解。
    """
    # 下界（弧长 >= 弦长）：Δθ_lower = L / |P'(θ)|
    slope = arc_len_derivative(theta_curr)
    dtheta_lo = max(L / max(slope, 1e-16), 1e-9)
    lo = theta_curr + dtheta_lo

    # 扩展上界直到弦长 >= L
    hi = theta_curr + 1.5 * dtheta_lo
    for _ in range(80):
        d = chord_distance(theta_curr, hi)
        if d >= L:
            break
        hi += 1.5 * dtheta_lo
    # 二分
    for _ in range(BISECT_MAX_IT):
        mid = 0.5 * (lo + hi)
        d = chord_distance(theta_curr, mid)
        if abs(d - L) < DIST_TOL:
            return mid
        if d < L:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -------------------- 链条构造与速度推导 --------------------

def build_theta_chain(theta_head: float) -> List[float]:
    """自龙头前把手（theta_head）起，依次求：
    [头前]、[第1身前]、...、[第221身前]、[尾前]、[尾后] 共 N_FRONT_HANDLES+1 个点。
    返回长度 N_FRONT_HANDLES+1 的 theta 列表。
    """
    thetas: List[float] = [theta_head]
    # 头前 -> 第1身前（等同于“头后”）
    thetas.append(solve_next_theta_by_chord(thetas[-1], L_HEAD))
    # 连续龙身（含龙尾前把手，均为 L_BODY）
    total_steps = N_FRONT_HANDLES - 1  # 余下要走的 L_BODY 步数（221 身 + 1 尾前）
    for _ in range(total_steps):
        thetas.append(solve_next_theta_by_chord(thetas[-1], L_BODY))
    # 再走一步得到尾后
    thetas.append(solve_next_theta_by_chord(thetas[-1], L_BODY))
    return thetas


def compute_positions(thetas: List[float]) -> List[Tuple[float, float]]:
    return [spiral_point(th) for th in thetas]


def compute_speeds(thetas: List[float]) -> Tuple[List[Tuple[float, float]], List[float]]:
    """根据投影等式逐节传递速度：
    v_i*(T_i·U_i) = v_{i+1}*(T_{i+1}·U_i)
    其中 T_i 为 theta 增大方向切向单位向量；实际盘入速度方向取 -T_i。
    返回：速度向量列表（vx,vy）、速度标量列表 speed
    """
    n = len(thetas)
    pts = compute_positions(thetas)
    T = [spiral_tangent_unit(th) for th in thetas]

    # 杆件方向 U_i：从 i 指向 i+1
    U: List[Tuple[float, float]] = []
    for i in range(n - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        norm = math.hypot(dx, dy)
        U.append((dx / norm, dy / norm))

    # 速度标量（沿曲线的大小）
    v_mag = [0.0] * n
    v_mag[0] = V_HEAD
    for i in range(n - 1):
        dot_i = T[i][0] * U[i][0] + T[i][1] * U[i][1]
        dot_ip1 = T[i + 1][0] * U[i][0] + T[i + 1][1] * U[i][1]
        # 数值保护，避免极小分母
        dot_ip1 = math.copysign(max(abs(dot_ip1), 1e-12), dot_ip1)
        v_mag[i + 1] = v_mag[i] * (dot_i / dot_ip1)
        # 绝对值（大小），方向统一用 -T_i 表示盘入
        v_mag[i + 1] = abs(v_mag[i + 1])

    # 速度向量（盘入方向 = -T）
    v_vec = [(-v_mag[i] * T[i][0], -v_mag[i] * T[i][1]) for i in range(n)]
    return v_vec, v_mag


# -------------------- 命名与导出 --------------------

def handle_names() -> List[str]:
    names: List[str] = []
    names.append("龙头")
    for i in range(1, N_BODY + 1):
        names.append(f"第{i}节龙身")
    names.append("龙尾（前）")
    names.append("龙尾（后）")
    return names


def compute_one_time(t: int) -> Dict[str, List]:
    """计算单个时刻 t 的所有把手位置与速度。返回打平的记录。"""
    # 龙头当前 theta：S(θ(t)) = S(θ0) - v*t
    S0 = arc_len_primitive(THETA0_HEAD)
    S_t = max(0.0, S0 - V_HEAD * t)
    theta_head_t = theta_from_arclen(S_t, THETA0_HEAD)

    thetas = build_theta_chain(theta_head_t)
    pts = compute_positions(thetas)
    v_vec, v_mag = compute_speeds(thetas)

    names = handle_names()
    # 打平
    recs = []
    for idx, name in enumerate(names):
        x, y = pts[idx]
        vx, vy = v_vec[idx]
        recs.append(
            {
                "t": t,
                "index": idx,
                "name": name,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "speed": v_mag[idx],
            }
        )
    return {"records": recs, "thetas": thetas}


def run_all() -> Tuple[pd.DataFrame, Dict[int, List[float]]]:
    all_recs: List[Dict] = []
    theta_map: Dict[int, List[float]] = {}
    for t in range(T_START, T_END + 1, DT):
        out = compute_one_time(t)
        all_recs.extend(out["records"])
        theta_map[t] = out["thetas"]
    df = pd.DataFrame(all_recs)
    return df, theta_map


def export_full(df: pd.DataFrame, path: str = "../data/result1.xlsx") -> None:
    # 统一排序
    df = df.sort_values(["t", "index"]).reset_index(drop=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="full")


def export_tables(df: pd.DataFrame, path: str = "../data/result_q1_tables.xlsx") -> None:
    """导出论文表1/表2所需的 7 个对象 × 5/7 个时刻。"""
    focus_times = [0, 60, 120, 180, 240, 300]
    # 目标对象索引
    # 龙头、龙身第1/51/101/151/201、龙尾后
    target_indices = [
        0,  # 龙头
        1,  # 第1节龙身
        51,  # 第51节龙身
        101,  # 第101节龙身
        151,  # 第151节龙身
        201,  # 第201节龙身
        N_FRONT_HANDLES,  # 龙尾（后） => 索引=前把手数
    ]

    names = handle_names()
    
    # 调试：检查数据范围
    print(f"数据时间范围: {df.t.min()} - {df.t.max()}")
    print(f"数据索引范围: {df.index.min()} - {df.index.max()}")
    print(f"目标索引: {target_indices}")
    print(f"名称列表长度: {len(names)}")

    # 位置表
    pos_rows = []
    for idx in target_indices:
        if idx >= len(names):
            print(f"警告：索引 {idx} 超出名称列表范围")
            continue
        row = {"对象": names[idx]}
        for tt in focus_times:
            mask = (df.t == tt) & (df.index == idx)
            filtered = df[mask]
            if len(filtered) == 0:
                print(f"警告：找不到时刻 {tt}，索引 {idx} 的数据")
                row[f"{tt}s x(m)"] = 0.0
                row[f"{tt}s y(m)"] = 0.0
            else:
                sub = filtered.iloc[0]
                row[f"{tt}s x(m)"] = round(sub.x, 6)
                row[f"{tt}s y(m)"] = round(sub.y, 6)
        pos_rows.append(row)
    pos_df = pd.DataFrame(pos_rows)

    # 速度表（大小）
    spd_rows = []
    for idx in target_indices:
        if idx >= len(names):
            continue
        row = {"对象": names[idx]}
        for tt in focus_times:
            mask = (df.t == tt) & (df.index == idx)
            filtered = df[mask]
            if len(filtered) == 0:
                row[f"{tt}s speed(m/s)"] = 0.0
            else:
                sub = filtered.iloc[0]
                row[f"{tt}s speed(m/s)"] = round(sub.speed, 6)
        spd_rows.append(row)
    spd_df = pd.DataFrame(spd_rows)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        pos_df.to_excel(writer, index=False, sheet_name="position")
        spd_df.to_excel(writer, index=False, sheet_name="speed")


def main():
    df, _ = run_all()
    export_full(df, path="d:/Files/Code/CUMCM2024/data/result1.xlsx")
    export_tables(df, path="d:/Files/Code/CUMCM2024/data/result_q1_tables.xlsx")


if __name__ == "__main__":
    main()
