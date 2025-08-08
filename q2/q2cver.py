# -*- coding: utf-8 -*-
"""
问题2：确定盘入终止时刻（避免板凳之间发生碰撞）
- 路径与几何同 q1：等距螺线 r=bθ，所有把手中心在螺线上；龙头以 1 m/s 沿弧长顺时针盘入。
- 从 t0=300 s 开始，以 1 s 步长推进，检测是否发生碰撞；一旦发现碰撞，用二分在 [t_safe, t_collide] 上逼近临界时刻。
- 板凳几何近似：每块板视为“胶囊体”（矩形+半圆端）：中心线为“同一板两孔连线”再各自向外延伸 0.275 m；半宽=0.15 m。
- 碰撞标准：任意两块非相邻的板，其中心线段之间的最小距离 < 2*0.15 m（考虑容差）即判定为碰撞。
- 输出：data/result2.xlsx（最终时刻所有把手 x,y,vx,vy,speed）与 data/result_q2_tables.xlsx（论文对象汇总）。
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict

import pandas as pd

# -------------------- 复用与常量（与 q1 保持一致） --------------------
PITCH = 0.55  # m
B = PITCH / (2.0 * math.pi)
V_HEAD = 1.0  # m/s
N_BODY = 221
L_HEAD = 3.41 - 0.55  # 2.86
L_BODY = 2.20 - 0.55  # 1.65
N_FRONT_HANDLES = 1 + N_BODY + 1  # 223（前把手数）
THETA0_HEAD = 2.0 * math.pi * 16.0

# 板宽与端部外伸
BOARD_HALF_W = 0.30 / 2.0  # 0.15 m（胶囊半径）
END_EXT = 0.275  # m（孔中心到板端）

# 数值参数
DIST_TOL = 1e-10
ANG_TOL = 1e-12
BISECT_MAX_IT = 80
NEWTON_MAX_IT = 50

# 时间推进
T0 = 300.0  # s，从此刻开始前进
DT_COARSE = 1.0  # s
DT_BISECTION_STOP = 1e-3  # s，二分停止阈值


# -------------------- 曲线与弧长工具（同 q1） --------------------

def spiral_point(theta: float) -> Tuple[float, float]:
    r = B * theta
    return r * math.cos(theta), r * math.sin(theta)


def spiral_tangent_unit(theta: float) -> Tuple[float, float]:
    cx, sx = math.cos(theta), math.sin(theta)
    tx = B * (cx - theta * sx)
    ty = B * (sx + theta * cx)
    norm = math.hypot(tx, ty)
    return (tx / norm, ty / norm)


def arc_len_primitive(theta: float) -> float:
    return 0.5 * B * (theta * math.sqrt(1.0 + theta * theta) + math.asinh(theta))


def arc_len_derivative(theta: float) -> float:
    return B * math.sqrt(1.0 + theta * theta)


def theta_from_arclen(target_S: float, theta_high: float) -> float:
    S0, SHigh = 0.0, arc_len_primitive(theta_high)
    if target_S <= S0:
        return 0.0
    if target_S >= SHigh:
        return theta_high
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
    slope = arc_len_derivative(theta_curr)
    dtheta_lo = max(L / max(slope, 1e-16), 1e-9)
    lo = theta_curr + dtheta_lo
    hi = theta_curr + 1.5 * dtheta_lo
    for _ in range(80):
        if chord_distance(theta_curr, hi) >= L:
            break
        hi += 1.5 * dtheta_lo
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


# -------------------- 链条与速度（同 q1 思路） --------------------

def build_theta_chain(theta_head: float) -> List[float]:
    thetas: List[float] = [theta_head]
    thetas.append(solve_next_theta_by_chord(thetas[-1], L_HEAD))
    total_steps = N_FRONT_HANDLES - 1
    for _ in range(total_steps):
        thetas.append(solve_next_theta_by_chord(thetas[-1], L_BODY))
    thetas.append(solve_next_theta_by_chord(thetas[-1], L_BODY))
    return thetas


def compute_positions(thetas: List[float]) -> List[Tuple[float, float]]:
    return [spiral_point(th) for th in thetas]


def compute_speeds(thetas: List[float]) -> Tuple[List[Tuple[float, float]], List[float]]:
    n = len(thetas)
    pts = compute_positions(thetas)
    T = [spiral_tangent_unit(th) for th in thetas]
    U: List[Tuple[float, float]] = []
    for i in range(n - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        norm = math.hypot(dx, dy)
        U.append((dx / norm, dy / norm))
    v_mag = [0.0] * n
    v_mag[0] = V_HEAD
    for i in range(n - 1):
        dot_i = T[i][0] * U[i][0] + T[i][1] * U[i][1]
        dot_ip1 = T[i + 1][0] * U[i][0] + T[i + 1][1] * U[i][1]
        dot_ip1 = math.copysign(max(abs(dot_ip1), 1e-12), dot_ip1)
        v_mag[i + 1] = abs(v_mag[i] * (dot_i / dot_ip1))
    v_vec = [(-v_mag[i] * T[i][0], -v_mag[i] * T[i][1]) for i in range(n)]
    return v_vec, v_mag


# -------------------- 矩形建模与碰撞检测 --------------------

def get_board_rectangles(points: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
    """把手点 → 每块板的四个顶点坐标
    返回矩形列表，长度等于板数量（223），第 i 个矩形由点 i 到 i+1 构成的中心线两端各外伸 END_EXT，宽度为 BOARD_HALF_W*2。
    每个矩形返回4个顶点：[左下, 右下, 右上, 左上]
    """
    rectangles: List[List[Tuple[float, float]]] = []
    for i in range(N_FRONT_HANDLES):  # 0..222 => 223 块板
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        # 中心线方向单位向量
        ux, uy = dx / L, dy / L
        # 垂直方向单位向量（左手系，向左为正）
        vx, vy = -uy, ux
        
        # 中心线两端外伸
        ax, ay = x0 - END_EXT * ux, y0 - END_EXT * uy
        bx, by = x1 + END_EXT * ux, y1 + END_EXT * uy
        
        # 四个顶点：从中心线向两侧扩展半宽
        # 左下 = a点向左偏移
        p1 = (ax + BOARD_HALF_W * vx, ay + BOARD_HALF_W * vy)
        # 右下 = b点向左偏移  
        p2 = (bx + BOARD_HALF_W * vx, by + BOARD_HALF_W * vy)
        # 右上 = b点向右偏移
        p3 = (bx - BOARD_HALF_W * vx, by - BOARD_HALF_W * vy)
        # 左上 = a点向右偏移
        p4 = (ax - BOARD_HALF_W * vx, ay - BOARD_HALF_W * vy)
        
        rectangles.append([p1, p2, p3, p4])
    return rectangles


def triangle_area(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """计算三角形面积（使用叉积公式）"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)


def point_in_rectangle(point: Tuple[float, float], rect_vertices: List[Tuple[float, float]]) -> bool:
    """使用面积法判断点是否在矩形内部
    如果点和矩形四个顶点构成的四个三角形面积之和等于矩形面积，则点在矩形内
    """
    px, py = point
    p1, p2, p3, p4 = rect_vertices
    
    # 计算矩形面积（使用两个三角形）
    rect_area = triangle_area(p1, p2, p3) + triangle_area(p1, p3, p4)
    
    # 计算四个三角形面积
    area1 = triangle_area(point, p1, p2)
    area2 = triangle_area(point, p2, p3)
    area3 = triangle_area(point, p3, p4)
    area4 = triangle_area(point, p4, p1)
    
    total_area = area1 + area2 + area3 + area4
    
    # 容差比较（考虑浮点误差）
    return abs(total_area - rect_area) < 1e-9


def has_collision(points: List[Tuple[float, float]]) -> bool:
    """检测板凳之间是否发生碰撞
    对每块板的四个顶点，检查是否在其他非相邻板的内部
    """
    rectangles = get_board_rectangles(points)
    n = len(rectangles)
    
    for i in range(n):
        rect_i = rectangles[i]
        # 检查当前板的四个顶点
        for vertex in rect_i:
            # 检查与所有其他板的碰撞
            for j in range(n):
                # 跳过自己和相邻板
                if abs(j - i) <= 1:
                    continue
                
                rect_j = rectangles[j]
                # 如果当前板的顶点在另一块板内部，则发生碰撞
                if point_in_rectangle(vertex, rect_j):
                    return True
    
    return False


# -------------------- 时间搜索与导出 --------------------

def compute_at_time(t: float) -> Dict:
    S0 = arc_len_primitive(THETA0_HEAD)
    S_t = max(0.0, S0 - V_HEAD * t)
    theta_head_t = theta_from_arclen(S_t, THETA0_HEAD)
    thetas = build_theta_chain(theta_head_t)
    pts = compute_positions(thetas)
    v_vec, v_mag = compute_speeds(thetas)
    return {"t": t, "thetas": thetas, "points": pts, "v_vec": v_vec, "v_mag": v_mag}


def find_termination_time() -> Dict:
    # 先验证 t0 无碰撞
    print("验证 t=300s 是否无碰撞...")
    state = compute_at_time(T0)
    if has_collision(state["points"]):
        print("警告: t=300s 已发生碰撞，请检查实现")
    else:
        print("t=300s 确认无碰撞，开始前进搜索...")

    # 粗步长前进，找到碰撞区间
    t_safe = T0
    state_safe = state
    t = T0
    step_count = 0
    while True:
        t_next = t + DT_COARSE
        next_state = compute_at_time(t_next)
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"已检查到 t = {t_next:.1f} s")
            
        if has_collision(next_state["points"]):
            print(f"在 t = {t_next:.1f} s 发现碰撞，开始二分逼近...")
            t_lo, st_lo = t_safe, state_safe
            t_hi, st_hi = t_next, next_state
            break
        t_safe, state_safe = t_next, next_state
        t = t_next
        # 保护：若龙头已到中心（θ=0），退出
        if theta_from_arclen(max(0.0, arc_len_primitive(THETA0_HEAD) - V_HEAD * t), THETA0_HEAD) <= 1e-9:
            print("龙头已到达螺线中心，未发生碰撞")
            return {"t": t, **next_state}

    # 二分逼近
    bisect_count = 0
    while t_hi - t_lo > DT_BISECTION_STOP:
        tm = 0.5 * (t_lo + t_hi)
        st_m = compute_at_time(tm)
        bisect_count += 1
        
        if bisect_count % 5 == 0:
            print(f"二分第 {bisect_count} 次: [{t_lo:.6f}, {t_hi:.6f}]")
            
        if has_collision(st_m["points"]):
            t_hi, st_hi = tm, st_m
        else:
            t_lo, st_lo = tm, st_m
    
    print(f"二分完成，最终安全时刻: t = {t_lo:.6f} s")
    # 终止时刻取不碰撞侧（最大安全时刻）
    return st_lo


def handle_names() -> List[str]:
    names: List[str] = []
    names.append("龙头")
    for i in range(1, N_BODY + 1):
        names.append(f"第{i}节龙身")
    names.append("龙尾（前）")
    names.append("龙尾（后）")
    return names


def export_result(state: Dict, path_full: str, path_tables: str) -> None:
    t = state["t"]
    pts = state["points"]
    v_vec, v_mag = state["v_vec"], state["v_mag"]
    names = handle_names()

    rows = []
    for idx, name in enumerate(names):
        x, y = pts[idx]
        vx, vy = v_vec[idx]
        rows.append({
            "t": round(t, 6),
            "index": idx,
            "name": name,
            "x": x, "y": y,
            "vx": vx, "vy": vy,
            "speed": v_mag[idx],
        })
    df = pd.DataFrame(rows).sort_values(["index"]).reset_index(drop=True)
    with pd.ExcelWriter(path_full, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="final")

    # 论文摘录（单时刻）
    target_indices = [0, 1, 51, 101, 151, 201, N_FRONT_HANDLES]
    pos_rows, spd_rows = [], []
    for idx in target_indices:
        sub = df[df["index"] == idx].iloc[0]
        pos_rows.append({"对象": names[idx], "x(m)": round(sub.x, 6), "y(m)": round(sub.y, 6)})
        spd_rows.append({"对象": names[idx], "speed(m/s)": round(sub.speed, 6)})
    pos_df = pd.DataFrame(pos_rows)
    spd_df = pd.DataFrame(spd_rows)
    with pd.ExcelWriter(path_tables, engine="xlsxwriter") as writer:
        pos_df.to_excel(writer, index=False, sheet_name="position")
        spd_df.to_excel(writer, index=False, sheet_name="speed")


def main():
    print("开始寻找盘入终止时刻...")
    final_state = find_termination_time()
    print(f"找到终止时刻: t = {final_state['t']:.6f} s")
    export_result(
        final_state,
        path_full="d:/Files/Code/CUMCM2024/data/result2.xlsx",
        path_tables="d:/Files/Code/CUMCM2024/data/result_q2_tables.xlsx",
    )
    print("结果已导出到 result2.xlsx 和 result_q2_tables.xlsx")


if __name__ == "__main__":
    main()
