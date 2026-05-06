"""
Example 6 — General Model: Pelicun (FEMA P-58) 地震损失评估（基于 IDA 结果，自动生成构件）

非结构构件数量由 NormQtyPact 自动生成（需要 Windows + Microsoft Excel）；
结构构件由用户手动定义。与 Example 7 的区别：本例通过 NormQtyPact 自动
推算非结构构件，Example 7 使用已有的完整 CMP_QNT.csv 文件直接驱动 pelicun。

工作流程:
  1. 读取 Example4_GeneralModel_IDA/Output/IDA_results.csv
  2. 从 IDA 结果中提取目标 IM 水平处各条地震波的 EDP 原始样本
  3. 将所有地震波 EDP 样本直接传入 PelicunLossAssessment（coupled_edp=True）
  4. 调用 PelicunLossAssessment.LossAssessment() 执行 FEMA P-58 概率性损失评估
  5. 输出修复费用、修复时间分布统计

模型说明:
  - 6 层钢矩形框架 (Steel Moment Frame)
  - 平面尺寸约 18m × 18m，楼层面积 324 m²/层 ≈ 3487 sqft
  - 使用功能：商业办公 (OFFICE)
  - 目标 Sa = 0.3g

构件 ID 参考（FEMA P-58 数据库）:
  B.10.41.001a  钢矩形框架梁柱节点（有侧向支撑）
  B.10.41.002a  钢矩形框架梁柱节点（无侧向支撑）
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

examples_dir = str(Path(__file__).resolve().parent.parent)
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

from MDOFModel.loss.PelicunLossAssessment import PelicunLossAssessment

# ── 模型参数 ─────────────────────────────────────────────────────────────────
NUM_OF_STORIES  = 6
FLOOR_AREA_M2   = 324.0                        # m²/层
FLOOR_AREA_SQFT = FLOOR_AREA_M2 * 10.764       # ≈ 3487 sqft/层
OCCUPANCY_TYPE  = 'OFFICE'

# 建筑替换费用（美元）
REPLACEMENT_COST = 5000000.0
REPLACEMENT_TIME = 365.0 * 250.0   # 倒塌/替换修复时间（工人·天），按实际项目调整

# ── IDA 结果路径（来自 Example 4）────────────────────────────────────────────
IDA_CSV = str(
    Path(__file__).resolve().parent / 'IDA_results.csv'
)

# ── 目标 Sa (g) ───────────────────────────────────────────────────────────────
IM_TARGET = 0.6     # g

# ── 结构构件定义 ──────────────────────────────────────────────────────────────
# 每层 8 个 SMF 节点（B.10.41.001a）
struct_cmp = PelicunLossAssessment.make_struct_cmp(
    cmp_id_list = ['B.10.41.001a'] * NUM_OF_STORIES,
    loc_list    = [str(i + 1) for i in range(NUM_OF_STORIES)],
    dir_list    = ['1'] * NUM_OF_STORIES,
    qty_list    = [8.0] * NUM_OF_STORIES,
    unit_list   = ['ea'] * NUM_OF_STORIES,
)

# 输出目录
CFDir = Path(__file__).resolve().parent / 'Output'
CFDir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    print('=' * 65)
    print('  Example 6: Pelicun (FEMA P-58) 损失评估（基于 IDA 结果）')
    print('=' * 65)
    print(f'  IDA 结果来源 : {IDA_CSV}')
    print(f'  目标 IM      : {IM_TARGET} g')
    print(f'  楼层数       : {NUM_OF_STORIES}')
    print(f'  使用类型     : {OCCUPANCY_TYPE}')
    print(f'  替换费用     : {REPLACEMENT_COST:,.0f} USD')
    print(f'  替换时间     : {REPLACEMENT_TIME:,.0f} 工人·天')
    print()

    # ── Step 1：初始化评估对象 ────────────────────────────────────────────
    la = PelicunLossAssessment(
        NumOfStories        = NUM_OF_STORIES,
        FloorArea_sqft      = FLOOR_AREA_SQFT,
        OccupancyType       = OCCUPANCY_TYPE,
        SampleSize          = 500,
        IrreparableMedian   = 0.01,
        IrreparableLogStd   = 0.3,
    )

    # ── Step 2 & 3：直接传入 IDA CSV，执行 Pelicun (FEMA P-58) 损失评估 ──
    # IdaCsv 会自动在 ImLevel 处插值提取 EDP，无需手动调用 interp_edp_from_ida
    results = la.LossAssessment(
        IdaCsv          = IDA_CSV,
        ImLevel         = IM_TARGET,
        StructuralCmp   = struct_cmp,
        ReplacementCost = REPLACEMENT_COST,
        ReplacementTime = REPLACEMENT_TIME,
        CollapseMedian  = 1.5,    # 倒塌易损性中值 Sa (g)，根据实际模型调整
        CollapseLogStd  = 0.4,
        OutputDir       = str(CFDir),
    )

    # ── Step 4：输出结果 ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  损失评估结果汇总（FEMA P-58 方法）')
    print('=' * 65)
    print(f'  平均修复费用 (USD)   : {results["MeanRepairCost"]:,.0f}')
    print(f'  修复费用标准差 (USD) : {results["StdRepairCost"]:,.0f}')
    if results['MeanRepairTime'] is not None:
        print(f'  平均修复时间 (工人·天): {results["MeanRepairTime"]:,.1f}')

    # 损失分布分位数
    agg = results['AggLoss']
    cost_col = PelicunLossAssessment._find_col(agg, 'Cost')
    if cost_col is not None:
        q = agg[cost_col].quantile([0.16, 0.50, 0.84])
        print(f'\n  修复费用分位数:')
        print(f'    16th 百分位: {q.iloc[0]:,.0f} USD')
        print(f'    50th 百分位: {q.iloc[1]:,.0f} USD')
        print(f'    84th 百分位: {q.iloc[2]:,.0f} USD')

    print(f'\n  输出目录: {CFDir}')
    print('=' * 65)
