"""
Example 5 — General Model Loss Assessment（基于 IDA_3D 结果）
==========================================================
使用双向地震动（FEMA P-695 记录对）进行 3D IDA 分析，取 X/Y 双向
EDP 包络（逐层最大值）后，根据 Hazus 方法进行地震损失评估。

工作流程:
  1. 若本地 IDA_results.csv 为 3D 格式，直接读取；否则运行 IDA_3D 分析
  2. 调用 IDA3D_to_2d_envelope 取双向包络，转为 Hazus 兼容格式
  3. 在各目标 IM 水平下模拟 EDP 样本
  4. 对每个 EDP 样本执行 Hazus 损失评估，得到损失分布
  5. 输出损失统计并保存 CSV

模型说明:
- 6 层钢矩形框架 (Steel Moment Frame, S1M)
- 平面尺寸约 18m × 18m，楼层面积 324 m²/层
- 按 moderate-code 设计等级
- 使用功能：商业办公 (COM4)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

examples_dir = str(Path(__file__).resolve().parent.parent)
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

from MDOFModel.analysis import IDA_2D
from MDOFModel.loss.Tool_LossAssess import Simulate_losses_given_IM_basedon_IDA

# ── 模型参数 ──────────────────────────────────────────────────────────────
NUM_OF_STORIES  = 6
FLOOR_AREA      = 324.0        # m²/层
STRUCTURAL_TYPE = 'S1'         # 钢矩形框架（自动匹配 S1M）
DESIGN_LEVEL    = 'moderate-code'
OCCUPANCY_CLASS = 'COM4'       # 商业办公
# DesignInfo 字典（Hazus 方法）
DESIGN_INFO = {'Code': 'Hazus', 'SeismicDesignLevel': DESIGN_LEVEL}

# ── IDA 结果路径 ───────────────────────────────────────────────────────────
IDA_CSV = Path(__file__).resolve().parent / 'IDA_results.csv'

# ── 目标 IM 与模拟参数 ────────────────────────────────────────────────────
IM_LIST = [0.1, 0.3, 0.5, 0.7, 1.0]   # 需要评估的 Sa 水平 (g)
N_SIM   = [500] * len(IM_LIST)         # 每个 IM 水平的蒙特卡洛模拟次数
BETA_M  = 0.25                         # 结构模型不确定性 β_M（对数标准差），用于放大 EDP 协方差

# 输出目录
CFDir = Path(__file__).resolve().parent / 'Output_new'
CFDir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    print('=' * 65)
    print('  Example 5: General Model — Hazus 损失评估（基于 3D IDA 结果）')
    print('=' * 65)

    # ── Step 1：取双向包络，转为 Hazus 兼容的 2D 等效 DataFrame ──────────
    ida_2d_envelope = IDA_2D.IDA_bidir_to_envelope(IDA_CSV)
    print(f'  IDA 结果：{len(ida_2d_envelope)} 行（X/Y 包络后）')
    print(f'  目标 IM 列表 : {IM_LIST} g')
    print(f'  每 IM 模拟数 : {N_SIM[0]}')
    print(f'  结构模型不确定 β_M : {BETA_M}')
    print()

    # ── Step 2：执行基于 IDA 的 EDP 模拟 + Hazus 损失评估 ──────────────
    SimEDP, df = Simulate_losses_given_IM_basedon_IDA(
        IDA_result    = ida_2d_envelope,   # 直接传入 2D 包络 DataFrame
        IM_list       = IM_LIST,
        N_Sim         = N_SIM,
        betaM         = BETA_M,
        OutputDir     = str(CFDir),
        NumofStories  = NUM_OF_STORIES,
        FloorArea     = FLOOR_AREA,
        StructuralType= STRUCTURAL_TYPE,
        DesignInfo    = DESIGN_INFO,
        OccupancyClass= OCCUPANCY_CLASS,
    )

    # ── 按 IM 汇总输出结果 ────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  损失评估结果汇总（按 IM 分组均值）')
    print('=' * 65)
    for im in IM_LIST:
        subset = df[df['IM'] == im]
        if len(subset) == 0:
            print(f'  IM = {im:.2f}g : 无模拟结果（IDA 数据覆盖范围不足）')
            continue
        mean_cost = subset['RepairCost_Total'].mean()
        std_cost  = subset['RepairCost_Total'].std()
        p50_cost  = subset['RepairCost_Total'].quantile(0.50)
        print(
            f'  IM = {im:.2f}g | '
            f'均值修复费用 = {mean_cost:,.0f} USD, '
            f'标准差 = {std_cost:,.0f} USD, '
            f'中位数 = {p50_cost:,.0f} USD'
        )

    print(f'\n  SimEDP 已保存至   : {CFDir / "SimEDP.csv"}')
    print(f'  BldLoss 已保存至  : {CFDir / "BldLoss.csv"}')
    print('=' * 65)
