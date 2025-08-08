#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 位置与速度计算（按图片推导的方程）
- 阿基米德螺线模型
- 链式刚性约束 + 速度传递
- 交互：可在命令行调整螺距、速度、时间范围与步长
输出：
- data/result_q1_tables.xlsx（论文表1/表2）
- data/result_q1_full.xlsx（每秒全节点 x,y,v）
"""
import math
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# --------- 数学工具：F(θ) 与其反解（对应公式(5)） ---------
def F_theta(theta: float) -> float:
    """F(θ) = 1/2[ θ*sqrt(1+θ^2) + ln(θ + sqrt(1+θ^2)) ]，θ>0"""
    s = math.sqrt(1.0 + theta * theta)
    return 0.5 * (theta * s + math.log(theta + s))


def inv_F(target: float, theta_init: float) -> float:
    """
    由 F(θ) = target 反解 θ。
    使用守护牛顿法：牛顿迭代，发散时回退二分。
    """
    # 构造一个包络区间 [lo, hi]，F 单调递增，θ>0
    # 以初值为中心扩展
    lo = max(1e-12, theta_init * 0.5)
    hi = max(theta_init * 1.5, theta_init + 1.0)
    # 扩展直到覆盖 target
    def ensure_bracket():
        nonlocal lo, hi
        Flo, Fhi = F_theta(lo), F_theta(hi)
        it = 0
        while not (Flo <= target <= Fhi) and it < 60:
            if target < Flo:
                lo *= 0.5
                Flo = F_theta(lo)
            else:
                hi *= 1.5
                Fhi = F_theta(hi)
            it += 1
    ensure_bracket()

    x = max(lo + 1e-9, min(theta_init, hi - 1e-9))
    for _ in range(30):
        Fx = F_theta(x)
        diff = Fx - target
        if abs(diff) < 1e-12:
            return x
        dFx = math.sqrt(1.0 + x * x)  # F'(θ) = sqrt(1+θ^2)
        x_new = x - diff / dFx
        if x_new <= lo or x_new >= hi or not math.isfinite(x_new):
            # 二分回退
            x_new = 0.5 * (lo + hi)
        if diff > 0:
            hi = x
        else:
            lo = x
        x = x_new
    return x  # 已在守护区间内，足够精度


# --------- 链条几何：距离方程(6) 与求 θ_{i+1} ---------
@dataclass
class SpiralChainParams:
    a: float                    # 螺距（m）
    v0: float                   # 龙头速度（m/s）
    L: List[float]              # 刚性段长度数组 L_i
    theta0_init: float          # 初始 θ0
    t0: float = 0.0


def chord_distance_on_spiral(a: float, theta_i: float, theta_j: float) -> float:
    """两点均在 ρ=(a/(2π))θ 的螺线上时的直线距离（由(6)反推的根式）"""
    factor = a / (2.0 * math.pi)
    term = theta_j * theta_j + theta_i * theta_i - 2.0 * theta_j * theta_i * math.cos(theta_j - theta_i)
    return abs(factor) * math.sqrt(max(0.0, term))


def solve_theta_next(a: float, theta_i: float, L_i: float, theta_next_guess: float) -> float:
    """
    由约束(6)：给定 θ_i 与 L_i，解 θ_{i+1} > θ_i。
    使用单调二分 + 少量牛顿加速（可选）。
    """
    def f(th):
        return chord_distance_on_spiral(a, theta_i, th) - L_i

    # 保证区间 [lo, hi]，使 f(lo)<=0, f(hi)>=0
    lo = max(theta_i + 1e-12, theta_next_guess - 1.0)
    hi = max(theta_next_guess, theta_i + 0.1)

    # 扩展上界直到越过
    val_lo = f(lo)
    val_hi = f(hi)
    it_expand = 0
    while val_hi < 0 and it_expand < 60:
        # 经验步长：按 Δθ ~ L_i*(2π/a)
        step = max(0.2, L_i * (2.0 * math.pi / a) * 0.5)
        hi += step
        val_hi = f(hi)
        it_expand += 1
    # 若 lo 超过零，则向回收缩
    while val_lo > 0 and it_expand < 120:
        lo = max(theta_i + 1e-12, lo - 0.5)
        val_lo = f(lo)
        it_expand += 1

    # 二分
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        vm = f(mid)
        if abs(vm) < 1e-12:
            return mid
        if vm < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------- 速度传递(7) 与节点速度(8) ---------
def dtheta0_dt(a: float, v0: float, theta0: float) -> float:
    return -(2.0 * math.pi * v0) / (a * math.sqrt(1.0 + theta0 * theta0))


def propagate_dtheta_dt(theta_i: float, theta_ip1: float, dtheta_i: float) -> float:
    """公式(7)"""
    delta = theta_ip1 - theta_i
    s = math.sin(delta)
    c = math.cos(delta)
    num = theta_ip1 * c - theta_i + theta_i * theta_ip1 * s
    den = theta_ip1 - theta_i * c + theta_i * theta_ip1 * s
    # 小量守护，避免除零
    if abs(den) < 1e-14:
        den = math.copysign(1e-14, den if den != 0 else 1.0)
    return (num / den) * dtheta_i


def speed_from_dtheta(a: float, theta: float, dtheta: float) -> float:
    return abs((a / (2.0 * math.pi)) * math.sqrt(1.0 + theta * theta) * dtheta)


# --------- 主计算 ---------
def compute_states(params: SpiralChainParams, t_list: List[float]) -> Dict[float, Dict[str, np.ndarray]]:
    """
    返回：每个 t -> {'theta': θ数组(0..N), 'x':..., 'y':..., 'v':...}
    """
    a, v0, L = params.a, params.v0, params.L
    N = len(L)  # 段数=节点数-1；节点数=N+1
    # θ0(t) 的常量（5）
    C = F_theta(params.theta0_init)  # F(θ0(0))
    k = (2.0 * math.pi * v0) / a

    results: Dict[float, Dict[str, np.ndarray]] = {}
    # 上一时刻的 θ 作为初值
    theta_prev = np.zeros(N + 1)

    for idx, t in enumerate(t_list):
        # 解 θ0(t)
        target = C - k * (t - params.t0)
        theta0_guess = params.theta0_init if idx == 0 else theta_prev[0]
        th0 = inv_F(target, theta0_guess)

        theta = np.zeros(N + 1)
        theta[0] = th0

        # 递推 θ_{i+1}
        for i in range(N):
            guess = theta_prev[i + 1] if idx > 0 else (theta[i] + max(0.3, L[i] * (2.0 * math.pi / a) * 0.4))
            theta[i + 1] = solve_theta_next(a, theta[i], L[i], guess)

        # 速度：先 dθ/dt，再 v
        dth = np.zeros_like(theta)
        dth[0] = dtheta0_dt(a, v0, theta[0])
        for i in range(N):
            dth[i + 1] = propagate_dtheta_dt(theta[i], theta[i + 1], dth[i])

        rho = (a / (2.0 * math.pi)) * theta
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        v = np.array([speed_from_dtheta(a, theta[i], dth[i]) for i in range(N + 1)])

        results[t] = {'theta': theta, 'x': x, 'y': y, 'v': v}
        theta_prev = theta

    return results


# --------- 导出表格（表1/表2） ---------
def export_tables(results: Dict[float, Dict[str, np.ndarray]], times_for_paper: List[float], out_path: str):
    # 关心的索引
    idx_head = 0
    idx_body = [1, 51, 101, 151, 201]
    idx_tail_rear = max(len(next(iter(results.values()))['x']) - 1, 0)

    # 位置表（表1）
    rows_pos = [
        ("龙头 x (m)", idx_head, 'x'),
        ("龙头 y (m)", idx_head, 'y'),
    ]
    for k in idx_body:
        rows_pos.append((f"第{k}节龙身 x (m)", k, 'x'))
        rows_pos.append((f"第{k}节龙身 y (m)", k, 'y'))
    rows_pos.append(("龙尾（后） x (m)", idx_tail_rear, 'x'))
    rows_pos.append(("龙尾（后） y (m)", idx_tail_rear, 'y'))

    # 速度表（表2）
    rows_vel = [("龙头 (m/s)", idx_head)]
    for k in idx_body:
        rows_vel.append((f"第{k}节龙身 (m/s)", k))
    rows_vel.append(("龙尾（后） (m/s)", idx_tail_rear))

    cols = [f"{int(t)} s" for t in times_for_paper]

    def build_df_pos():
        data = []
        index_labels = []
        for name, idx, field in rows_pos:
            index_labels.append(name)
            row = []
            for t in times_for_paper:
                val = results[t][field][idx]
                row.append(round(float(val), 6))
            data.append(row)
        return pd.DataFrame(data, index=index_labels, columns=cols)

    def build_df_vel():
        data = []
        index_labels = []
        for name, idx in rows_vel:
            index_labels.append(name)
            row = []
            for t in times_for_paper:
                val = results[t]['v'][idx]
                row.append(round(float(val), 6))
            data.append(row)
        return pd.DataFrame(data, index=index_labels, columns=cols)

    df_pos = build_df_pos()
    df_vel = build_df_vel()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_pos.to_excel(writer, sheet_name="位置-表1")
        df_vel.to_excel(writer, sheet_name="速度-表2")


def export_full(results: Dict[float, Dict[str, np.ndarray]], out_path: str):
    """
    导出每秒全节点数据，按时间列排列的格式（龙头、各节龙身的x,y坐标）
    """
    # 获取所有时间点并排序
    time_points = sorted(results.keys())
    
    # 准备列名：每个时间点一列
    columns = [f"{int(t)} s" for t in time_points]
    
    # 准备行名和数据
    rows_data = []
    row_labels = []
    
    # 获取节点总数
    first_result = next(iter(results.values()))
    N = len(first_result['x'])
    
    # 龙头
    row_labels.append("龙头x (m)")
    row_data = []
    for t in time_points:
        val = results[t]['x'][0]  # 龙头是索引0
        row_data.append(round(float(val), 6))
    rows_data.append(row_data)
    
    row_labels.append("龙头y (m)")
    row_data = []
    for t in time_points:
        val = results[t]['y'][0]
        row_data.append(round(float(val), 6))
    rows_data.append(row_data)
    
    # 各节龙身（索引1到221）
    for i in range(1, min(222, N)):  # 龙身节点
        # x坐标
        row_labels.append(f"第{i}节龙身x (m)")
        row_data = []
        for t in time_points:
            val = results[t]['x'][i]
            row_data.append(round(float(val), 6))
        rows_data.append(row_data)
        
        # y坐标
        row_labels.append(f"第{i}节龙身y (m)")
        row_data = []
        for t in time_points:
            val = results[t]['y'][i]
            row_data.append(round(float(val), 6))
        rows_data.append(row_data)
    
    # 龙尾前把手（如果存在，索引222）
    if N > 222:
        row_labels.append("龙尾x (m)")
        row_data = []
        for t in time_points:
            val = results[t]['x'][222]
            row_data.append(round(float(val), 6))
        rows_data.append(row_data)
        
        row_labels.append("龙尾y (m)")
        row_data = []
        for t in time_points:
            val = results[t]['y'][222]
            row_data.append(round(float(val), 6))
        rows_data.append(row_data)
    
    # 龙尾后把手（如果存在，索引223）
    if N > 223:
        row_labels.append("龙尾（后）x (m)")
        row_data = []
        for t in time_points:
            val = results[t]['x'][223]
            row_data.append(round(float(val), 6))
        rows_data.append(row_data)
        
        row_labels.append("龙尾（后）y (m)")
        row_data = []
        for t in time_points:
            val = results[t]['y'][223]
            row_data.append(round(float(val), 6))
        rows_data.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(rows_data, index=row_labels, columns=columns)
    
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="full")


# --------- 交互与入口 ---------
def build_default_L() -> List[float]:
    """
    段长 L_i：
    L_0 = 2.86 (头凳)
    L_1..L_221 = 1.65 (龙身共 221 段)
    L_222 = 1.65 (龙尾前→后)
    共 223 段 -> 224 节点
    """
    L = [2.86] + [1.65] * 221 + [1.65]
    return L


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUMCM2024 A-板凳龙 Q1（位置与速度）")
    p.add_argument("--a", type=float, default=0.55, help="螺距 a (m)，默认 0.55")
    p.add_argument("--v0", type=float, default=1.0, help="龙头速度 v0 (m/s)，默认 1.0")
    p.add_argument("--t_end", type=float, default=300.0, help="结束时间(s)，默认 300")
    p.add_argument("--dt", type=float, default=1.0, help="时间步长(s)，默认 1")
    p.add_argument("--paper_times", type=str, default="0,60,120,180,240,300",
                   help="论文表格时间点(秒,逗号分隔)")
    p.add_argument("--no_full", action="store_true", help="不导出每秒全量表")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # 交互提示（可直接回车使用默认）
    print("参数设置（直接回车使用默认）：")
    try:
        a_in = input(f"- 螺距 a (m) [{args.a}]: ").strip()
        v0_in = input(f"- 龙头速度 v0 (m/s) [{args.v0}]: ").strip()
        t_end_in = input(f"- 结束时间 t_end (s) [{args.t_end}]: ").strip()
        dt_in = input(f"- 时间步长 dt (s) [{args.dt}]: ").strip()
        paper_times_in = input(f"- 论文表时间点(逗号) [{args.paper_times}]: ").strip()

        a = float(a_in) if a_in else args.a
        v0 = float(v0_in) if v0_in else args.v0
        t_end = float(t_end_in) if t_end_in else args.t_end
        dt = float(dt_in) if dt_in else args.dt
        paper_times = [float(s) for s in (paper_times_in if paper_times_in else args.paper_times).split(",")]
    except Exception as e:
        print(f"输入解析失败，使用命令行参数。原因：{e}")
        a, v0, t_end, dt = args.a, args.v0, args.t_end, args.dt
        paper_times = [float(s) for s in args.paper_times.split(",")]

    # 基本检查
    if dt <= 0:
        dt = 1.0
    if t_end < 0:
        t_end = 300.0

    L = build_default_L()
    theta0_init = 2.0 * math.pi * 16.0

    params = SpiralChainParams(a=a, v0=v0, L=L, theta0_init=theta0_init)
    t_list = [round(float(t), 10) for t in np.arange(0.0, t_end + 1e-9, dt).tolist()]
    print(f"开始计算：a={a}, v0={v0}, T=[0,{t_end}] s, dt={dt}, 段数={len(L)} ...")

    results = compute_states(params, t_list)

    # 导出论文表
    out_tables = r"d:\Files\Code\CUMCM2024\data\result_q1_tables.xlsx"
    export_tables(results, [float(t) for t in paper_times], out_tables)
    print(f"已导出论文表到：{out_tables}")

    # 可选：导出全量
    if not args.no_full:
        out_full = r"d:\Files\Code\CUMCM2024\data\result_q1_full.xlsx"
        export_full(results, out_full)
        print(f"已导出全量表到：{out_full}")

    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))