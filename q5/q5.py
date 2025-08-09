# -*- coding: utf-8 -*-
"""
Q5: 在 Q4 的路径上，龙头以恒速 v 行进；求使全队各把手速度均不超过 2 m/s 的最大龙头速度。
实现思路：
- 动态加载 q4/q4.py，复用其几何构造（先大圆弧 2a，再小圆弧 a）、统一路径与构链/速度投影。
- 先用 v=1 m/s 在 t∈[-100,100] 的整秒扫描得到全队的最大速度 smax1 及大致时刻区间；
- 在该 2 s 区间内做细化采样（密度可调）得到更精确的 smax1；
- 线性缩放估计 v* = 2 / smax1；随后用更密的时间采样校验（必要时对 v 做二分调整），输出最终 v*。
"""
from __future__ import annotations
import importlib.util
import math
import os
from typing import Tuple, List, Dict, Any

# -------------------- 动态加载 q4 模块 --------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
Q4_PATH = os.path.join(ROOT, 'q4', 'q4.py')

spec = importlib.util.spec_from_file_location('q4_runtime', Q4_PATH)
q4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q4)  # type: ignore

# -------------------- 构造 Q4 路径几何 --------------------
PITCH = 1.70
R_TURN = 9.0 / 2.0

def build_path() -> Any:
    B = q4.B_from_pitch(PITCH)
    theta_b = R_TURN / B
    # 设置旋转（左下象限 ≈ 225°）
    ang0 = theta_b % (2 * math.pi)
    target = 225.0 / 180.0 * math.pi
    q4.PHI = target - ang0
    # 搜索两段圆弧
    tg, a_best = q4.search_min_turn(theta_b, B)
    path = q4.CompositePath(tg)
    print(f"[Q5] 使用 Q4 几何：a={a_best:.6f}, Lturn={path.Lturn:.6f}")
    return path

# -------------------- 评价给定龙头速度的全局最大速度 --------------------

def max_speed_on_interval(path: Any, v_head: float, t_lo: float, t_hi: float, step: float) -> Tuple[float, float, int]:
    """返回 (max_speed, arg_t, arg_index)。step 可以为非整数。"""
    q4.V_HEAD = v_head
    t = t_lo
    best = -1.0
    best_t = t_lo
    best_idx = 0
    while t <= t_hi + 1e-9:
        s_head = v_head * t
        s_list, _pts = q4.build_chain_on_path(path, s_head)
        _v_vec, v_mag = q4.compute_speeds_on_path(path, s_list)
        vmax = max(v_mag)
        if vmax > best:
            best = vmax
            best_t = t
            best_idx = int(max(range(len(v_mag)), key=lambda i: v_mag[i]))
        t += step
    return best, best_t, best_idx


def refine_peak_time(path: Any, t_center: float, v_head: float, window: float = 1.0) -> Tuple[float, float]:
    """在 [t_center-1, t_center+1] 内做细化采样，返回 (best_t, best_speed)。"""
    lo = t_center - window
    hi = t_center + window
    # 采用等距细采样
    best, best_t, _ = max_speed_on_interval(path, v_head, lo, hi, step=0.05)
    return best_t, best

# -------------------- 主过程 --------------------

def main():
    path = build_path()

    # 1) v=1 m/s 粗扫整秒，定位峰值所在 2s 区间
    v1 = 1.0
    smax1, t_star, idx_star = max_speed_on_interval(path, v1, -100.0, 100.0, step=1.0)
    print(f"[Q5] v=1m/s 时最大速度≈{smax1:.6f} m/s，出现在 t≈{t_star:.3f}s，index≈{idx_star}")

    # 2) 区间细化
    t_peak, smax1_ref = refine_peak_time(path, t_star, v1, window=1.0)
    print(f"[Q5] 细化后峰值速度≈{smax1_ref:.6f} m/s，t≈{t_peak:.3f}s")
    if smax1_ref <= 0:
        raise RuntimeError('速度评估异常：最大值 <= 0')

    # 3) 线性缩放估计最大龙头速度
    v_est = 2.0 / smax1_ref
    print(f"[Q5] 线性缩放估计龙头最大恒速 v*≈{v_est:.6f} m/s")

    # 4) 校验：用更密时间采样检查是否所有把手速度≤2 m/s
    def feasible(v: float) -> bool:
        smax, _, _ = max_speed_on_interval(path, v, -100.0, 100.0, step=0.25)
        return smax <= 2.0 + 1e-9

    if feasible(v_est):
        v_final = v_est
    else:
        # 5) 不可行则对 v 做二分收敛
        lo, hi = 0.0, v_est
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        v_final = 0.5 * (lo + hi)

    print(f"[Q5] 满足约束（-100..100s 内各把手≤2 m/s）的最大龙头恒速：{v_final:.6f} m/s")

    # 可选：导出到 data/result5.txt
    out_dir = os.path.join(ROOT, 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'result5.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"v_max = {v_final:.6f} m/s\n")
    print(f"结果已写入 {out_path}")


if __name__ == '__main__':
    main()
