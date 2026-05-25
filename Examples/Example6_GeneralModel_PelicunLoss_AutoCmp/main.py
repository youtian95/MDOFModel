"""
Example 6 — General Model: Pelicun (FEMA P-58) 地震损失评估（基于 IDA_3D 结果，自动生成构件）

使用双向地震动（FEMA P-695 记录对）进行 3D IDA 分析，然后通过 PelicunLossAssessment
自动识别 3D CSV 格式，将 X 和 Y 双向 EDP 分别写入需求 CSV 的方向 1（X）和方向 2（Y）。

非结构构件数量由 NormQtyPact 自动生成（需要 Windows + Microsoft Excel）；
结构构件由用户手动定义。

工作流程:
  1. 读取本地 IDA_results.csv（3D 格式，含 MaxDrift_X/Y 等双向 EDP）
  2. CollapseAnalysis 自动识别 3D CSV，拟合倒塌易损性并筛选记录
  3. 调用 PelicunLossAssessment.LossAssessment(IdaCsv=...) 执行评估（自动识别 3D 格式）
  4. 输出修复费用、修复时间、倒塌概率和不可修复概率

模型说明:
  - 6 层钢矩形框架 (Steel Moment Frame)
  - 平面尺寸约 18m × 18m，楼层面积 324 m²/层 ≈ 3487 sqft
  - 使用功能：商业办公 (OFFICE)
  - 目标 Sa_gm = 2.0g

构件 ID 参考（FEMA P-58 数据库）:
  B.10.41.001a  钢矩形框架梁柱节点（有侧向支撑）
  B.10.41.002a  钢矩形框架梁柱节点（无侧向支撑）
"""

from pathlib import Path
import sys
import pandas as pd

examples_dir = str(Path(__file__).resolve().parent.parent)
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

from MDOFModel.loss.PelicunLossAssessment import PelicunLossAssessment
from MDOFModel.analysis import Collapse

# ── 模型参数 ─────────────────────────────────────────────────────────────────
NUM_OF_STORIES  = 6
FLOOR_AREA_M2   = 324.0                        # m²/层
FLOOR_AREA_SQFT = FLOOR_AREA_M2 * 10.764       # ≈ 3487 sqft/层
OCCUPANCY_TYPE  = 'OFFICE'
BUILDING_TYPE   = 'S1L'    # Hazus 结构类型代码，用于查询倒塌位移角阈值

# 建筑替换费用（美元）
REPLACEMENT_COST = 5000000.0
REPLACEMENT_TIME = 365.0 * 250.0   # 倒塌/替换修复时间（工人·天），按实际项目调整

# ── IDA 结果路径 ──────────────────────────────────────────────────────────────
IDA_CSV = Path(__file__).resolve().parent / 'IDA_results.csv'

# ── 目标 Sa_gm (g) ────────────────────────────────────────────────────────────
IM_TARGET =0.5     # g

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
CFDir = Path(__file__).resolve().parent / 'Output_new'
CFDir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    print('=' * 65)
    print('  Example 6: Pelicun (FEMA P-58) 损失评估（基于 3D IDA 结果）')
    print('=' * 65)
    print(f'  IDA 结果路径 : {IDA_CSV}')
    print(f'  目标 IM      : {IM_TARGET} g  (几何均值 Sa_gm)')
    print(f'  楼层数       : {NUM_OF_STORIES}')
    print(f'  使用类型     : {OCCUPANCY_TYPE}')
    print(f'  替换费用     : {REPLACEMENT_COST:,.0f} USD')
    print(f'  替换时间     : {REPLACEMENT_TIME:,.0f} 工人·天')
    print()

    # ── Step 1：倒塌分析（自动识别 3D 格式）──────────────────────────────
    ca = Collapse.CollapseAnalysis(IDA_CSV, building_type=BUILDING_TYPE)
    collapse_result = ca.fit_collapse_fragility(fig_path=str(CFDir / 'collapse_fragility.jpg'))
    collapse_median = collapse_result['median']
    collapse_logstd = collapse_result['logstd']
    print(f'  倒塌易损性中值 Sa_gm: {collapse_median:.3f} g')
    print(f'  倒塌易损性对数标准差: {collapse_logstd:.3f}')

    # 筛选非倒塌记录，保存过滤后的 3D IDA CSV
    filtered_df = ca.filter_collapse()
    filtered_csv = str(CFDir / 'IDA_results_filtered.csv')
    filtered_df.to_csv(filtered_csv, index=False, encoding='utf-8-sig')

    # ── Step 3：初始化评估对象 ────────────────────────────────────────────
    la = PelicunLossAssessment(
        NumOfStories        = NUM_OF_STORIES,
        FloorArea_sqft      = FLOOR_AREA_SQFT,
        OccupancyType       = OCCUPANCY_TYPE
    )

    # ── Step 4：执行 Pelicun (FEMA P-58) 损失评估 ────────────────────────
    # IdaCsv 自动识别 3D CSV：提取 X/Y 双向 EDP，分别对应需求 CSV 中的方向 1 和方向 2
    results = la.LossAssessment(
        IdaCsv          = filtered_csv,
        ImLevel         = IM_TARGET,
        StructuralCmp   = struct_cmp,
        ReplacementCost = REPLACEMENT_COST,
        ReplacementTime = REPLACEMENT_TIME,
        CollapseMedian  = collapse_median,
        CollapseLogStd  = collapse_logstd,
        OutputDir       = str(CFDir),
    )

    # ── Step 5：输出结果 ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  损失评估结果汇总（FEMA P-58 方法）')
    print('=' * 65)
    print(f'  倒塌概率: {results["CollapseProb"]:.4f}  ({results["CollapseProb"]*100:.2f}%)')
    print(f'  不可修复概率: {results["IrreparableProb"]:.4f}  ({results["IrreparableProb"]*100:.2f}%)')
    print(f'  平均修复费用 (USD): {results["MeanRepairCost"]:,.0f}')
    print(f'  修复费用标准差 (USD): {results["StdRepairCost"]:,.0f}')
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
