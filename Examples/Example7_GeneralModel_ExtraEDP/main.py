"""
Example 7 — 6 层钢矩形框架：自定义 EDP（柱截面应变）IDA 与 Pelicun 损失评估

使用复杂 6 层钢框架模型（Example_6Story_MRF_Model.py），演示以下完整流程：

  1. **GeneralModelWrapper 回调接口**：
     - ``extra_recorder_setup(tmp_dir)``：在每次动力分析开始前为各层代表性中柱
       注册 EnvelopeElement 截面变形 recorder（轴向应变 + 曲率），
       输出到进程唯一的临时目录 ``tmp_dir``。
     - ``extra_post_process(model, tmp_dir)``：在 ``ops.wipe()`` 后读取包络结果，
       按 ``极端纤维应变 ≈ |ε_axial| + |κ| × (d/2)`` 估算各层柱底最大应变，
       赋值给 ``model.MaxColStrain``。

  2. **3D IDA 分析**：使用双向地震动分量（X/Y 对），`ExtraEDP = {'STRAIN': 'MaxColStrain'}`
     将柱应变写入 IDA CSV（``STRAIN_X`` / ``STRAIN_Y`` 两列）。
     为节约时间，仅使用前 5 对 FEMA P-695 远场记录和 2 个 IM 强度级别。

  3. **自定义构件**：使用 ``make_custom_cmp`` 定义以 ``STRAIN`` 为 EDP 的柱损伤
     构件（DS1: 屈服应变 0.003，DS2: 显著塑性 0.010），与标准 SMF 节点构件
     一起传入 Pelicun 损失评估。

  4. **损失评估**：目标 IM = 0.6 g（位于 3D IDA 插值区间 [0.4, 0.8] 内），
     执行 FEMA P-58 方法并汇总修复费用与时间。

模型参数（来自 Example_6Story_MRF_Model.py）：
  - 6 层钢矩形框架，柱 W650/W550/W450，梁 W450/W400
  - Y 方向层高：5000-4000-4000-4000-4000-4000 mm
  - 楼层面积 324 m²，单位体系 mm-N-s，g = 9800 mm/s²

并行说明
--------
``extra_recorder_setup`` / ``extra_post_process`` 使用模块顶层定义的可调用 dataclass
（``ColStrainRecorder`` / ``ColStrainPostProcess``）实现。定义在模块顶层，没有放在
``if __name__ == '__main__':`` 块内，Python ``multiprocessing`` spawn 模式对它们可
正确 pickle，因此 ``NumPool > 1`` 的并行 IDA 完全受支持。

``tmp_dir``（= ``TmpDir/opensees_{UniqueRecorderPrefix}``）在每条地震波记录
对应唯一子目录，确保并行进程间 recorder 输出文件不冲突。
"""

from pathlib import Path
import sys
from dataclasses import dataclass
import openseespy.opensees as ops
import numpy as np
import pandas as pd

_examples_dir = str(Path(__file__).resolve().parent.parent)
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)

from Example_6Story_MRF_Model import BASE_NODES, FLOOR_NODES, STORY_HEIGHTS, build_model

from MDOFModel.models.GeneralModelWrapper import GeneralModelWrapper
from MDOFModel.analysis.IDA_2D import IDAAnalysis
from MDOFModel.loss.PelicunLossAssessment import PelicunLossAssessment
from MDOFModel.analysis import Collapse

# ── 模型参数 ──────────────────────────────────────────────────────────────────
NUM_OF_STORIES  = 6
FLOOR_AREA_M2   = 324.0
FLOOR_AREA_SQFT = FLOOR_AREA_M2 * 10.764        # ≈ 3487 sqft/层
OCCUPANCY_TYPE  = 'OFFICE'
BUILDING_TYPE   = 'S1L'

REPLACEMENT_COST = 5000000.0   # USD
REPLACEMENT_TIME = 365.0 * 250.0  # worker·day

IM_LIST   = [0.4, 0.8]   # 2 个 IM 级别（节约运行时间）
IM_TARGET = 0.6          # 损失评估目标 IM，位于 IDA 插值区间内

# ── 柱应变 recorder 参数 ──────────────────────────────────────────────────
COL_ELEMENTS = (10103, 10203, 10303, 10403, 10503, 10603)  # 各层代表性中柱单元编号
COL_HALF_D   = (325.0, 325.0, 275.0, 225.0, 225.0, 225.0) # 截面半高 d/2 (mm)


@dataclass(frozen=True)
class ColStrainRecorder:
    """EnvelopeElement 截面变形 recorder 注册器（模块顶层定义，支持 pickle）。"""

    element_tags: tuple[int, ...]

    def __call__(self, tmp_dir: Path) -> None:
        for i, ele_tag in enumerate(self.element_tags):
            ops.recorder(
                'EnvelopeElement', '-file',
                str(tmp_dir / f'col_deform_{i}.out'),
                '-ele', ele_tag, 'section', 1, 'deformation',
            )


@dataclass(frozen=True)
class ColStrainPostProcess:
    """EnvelopeElement 包络结果读取器（模块顶层定义，支持 pickle）。

    极端纤维应变估算： ``ε_extreme ≈ |ε_axial|_absMax + |κ|_absMax × (d/2)``
    """

    half_d: tuple[float, ...]  # 各层截面半高 d/2 (mm)

    def __call__(self, model, tmp_dir: Path) -> None:
        strains = []
        for i, hd in enumerate(self.half_d):
            try:
                arr = np.atleast_2d(np.loadtxt(tmp_dir / f'col_deform_{i}.out'))
                strains.append(
                    float(arr[2, 0]) + float(arr[2, 1]) * hd
                    if arr.shape[0] >= 3 and arr.shape[1] >= 2
                    else float('nan')
                )
            except Exception:
                strains.append(float('nan'))
        model.MaxColStrain = strains

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent / 'Output_new'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IDA_CSV      = OUT_DIR / 'IDA_results.csv'
FILTERED_CSV = OUT_DIR / 'IDA_results_filtered.csv'


# 回调实例（模块顶层，将被并行工作进程 pickle）
_col_strain_recorder     = ColStrainRecorder(COL_ELEMENTS)
_col_strain_post_process = ColStrainPostProcess(COL_HALF_D)

# ─────────────────────────────────────────────────────────────────────────────
# 构件定义
# ─────────────────────────────────────────────────────────────────────────────

# 标准 FEMA P-58 SMF 梁柱节点（B.10.41.001a），每层 8 个节点
struct_cmp = PelicunLossAssessment.make_struct_cmp(
    cmp_id_list = ['B.10.41.001a'] * NUM_OF_STORIES,
    loc_list    = [str(i + 1) for i in range(NUM_OF_STORIES)],
    dir_list    = ['1'] * NUM_OF_STORIES,
    qty_list    = [8.0] * NUM_OF_STORIES,
    unit_list   = ['ea'] * NUM_OF_STORIES,
)

# 自定义构件：柱截面钢材应变损伤（Custom.ColStrain.001）
#
#   EDP 类型  : 'STRAIN'  ← 与 ExtraEDP = {'STRAIN': 'MaxColStrain'} 的键一致
#   数量       : 每层 5 根柱（col=1..5）
#
#   损伤状态（来源：ATC-58 / 工程判断）：
#     DS1 (初始屈服)   ε ≥ 0.003  (ε_y = Fy/Es = 460/200000 ≈ 0.0023，含轴力余量)
#     DS2 (显著塑性)   ε ≥ 0.010
#
#   修复费用与时间为示例参考值（2011 USD，worker·day）。
custom_cmp = PelicunLossAssessment.make_custom_cmp(
    cmp_id_list   = ['Custom.ColStrain.001'] * NUM_OF_STORIES,
    edp_type_list = ['STRAIN']               * NUM_OF_STORIES,
    loc_list      = [str(i + 1) for i in range(NUM_OF_STORIES)],
    dir_list      = ['1'] * NUM_OF_STORIES,
    qty_list      = [5.0] * NUM_OF_STORIES,
    frag_theta_0  = [[0.003, 0.010]] * NUM_OF_STORIES,
    frag_theta_1  = [[0.40,  0.40]]  * NUM_OF_STORIES,
    cost_theta_0  = [[8_000, 30_000]] * NUM_OF_STORIES,
    cost_theta_1  = [[0.40,  0.40]]   * NUM_OF_STORIES,
    time_theta_0  = [[2.0,   10.0]]   * NUM_OF_STORIES,
    time_theta_1  = [[0.40,  0.40]]   * NUM_OF_STORIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print('=' * 65)
    print('  Example 7: 6 层 MRF — 柱应变 (STRAIN) ExtraEDP + 损失评估')
    print('=' * 65)

    # ── Step 1：初始化 GeneralModelWrapper ───────────────────────────────────
    print('\n[Step 1] 初始化结构模型 (GeneralModelWrapper)...')
    wrapper = GeneralModelWrapper(
        build_model_func = build_model,
        floor_nodes      = FLOOR_NODES,
        story_heights    = STORY_HEIGHTS,
        base_nodes       = BASE_NODES,
        dof              = 1,
    )
    # 挂载柱应变 recorder 回调（在标准 recorder 注册完毕后追加）
    wrapper.extra_recorder_setup = _col_strain_recorder
    wrapper.extra_post_process   = _col_strain_post_process
    print(f'  基本周期 T1 = {wrapper.T1:.3f} s')

    # ── Step 2：选取前 5 对 FEMA P-695 远场记录（X/Y 分量对）────────────────
    print('\n[Step 2] 选取地震动记录对...')
    from MDOFModel import __file__ as _mdof_file
    fema_dir = Path(_mdof_file).parent / 'Resources' / 'FEMA_P-695_far-field_ground_motions'
    meta_df  = pd.read_table(str(fema_dir / 'MetaData.txt'), sep=',')
    pairs_5 = [
        (str(fema_dir / x.replace('.txt', '')),
         str(fema_dir / y.replace('.txt', '')))
        for x, y in zip(
            meta_df['AccelXfile'].head(5).tolist(),
            meta_df['AccelYfile'].head(5).tolist(),
        )
    ]
    print(f'  已选 {len(pairs_5)} 对记录，IM 级别 = {IM_LIST} g')

    # ── Step 3：3D IDA 分析 ──────────────────────────────────────────────────
    # 回调定义在独立模块 mrf_strain_callbacks.py，可被 multiprocessing 正确
    # pickle → NumPool > 1 的并行执行完全支持。
    # tmp_dir = TmpDir/opensees_{UniqueRecorderPrefix} 已保证各进程文件不冲突。
    # 3D IDA：对每对记录分别运行 X、Y 方向动力分析，IM 取两分量谱加速度的几何均值。
    print('\n[Step 3] 执行 3D IDA（2 个 IM × 5 对记录 × 2 方向 = 20 次动力分析）...')
    ida = IDAAnalysis(wrapper)
    ida_result = ida.Analyze(
        IM_list  = IM_LIST,
        records  = pairs_5,
        ExtraEDP = {'STRAIN': 'MaxColStrain'},
        NumPool  = 5,
        DeltaT   = 0.2
    )
    ida.SaveToCSV(str(IDA_CSV))
    print(f'  3D IDA 完成，共 {len(ida_result)} 行结果 → {IDA_CSV.name}')
    if 'STRAIN_X' in ida_result.columns:
        print(f'  STRAIN_X 列写入成功，第 1 行示例值: {ida_result["STRAIN_X"].iloc[0]}')

    # ── Step 4：倒塌分析与记录过滤 ───────────────────────────────────────────
    print('\n[Step 4] 倒塌分析...')
    ca = Collapse.CollapseAnalysis(str(IDA_CSV), building_type=BUILDING_TYPE)
    collapse_result = ca.fit_collapse_fragility(
        fig_path=str(OUT_DIR / 'collapse_fragility.jpg')
    )
    collapse_median = collapse_result['median']
    collapse_logstd = collapse_result['logstd']
    print(f'  倒塌中值 Sa = {collapse_median:.3f} g，β = {collapse_logstd:.3f}')

    filtered_df = ca.filter_collapse()
    filtered_df.to_csv(str(FILTERED_CSV), index=False, encoding='utf-8-sig')
    print(f'  过滤后保留 {len(filtered_df)} 行记录')

    # ── Step 5：Pelicun (FEMA P-58) 损失评估 ─────────────────────────────────
    # demand.csv 中将自动出现 "1-STRAIN-{层}-1" 和 "1-STRAIN-{层}-2" 列，
    # 分别来自 STRAIN_X / STRAIN_Y，供 Custom.ColStrain.001 使用
    print('\n[Step 5] 执行 Pelicun 损失评估...')
    la = PelicunLossAssessment(
        NumOfStories   = NUM_OF_STORIES,
        FloorArea_sqft = FLOOR_AREA_SQFT,
        OccupancyType  = OCCUPANCY_TYPE,
    )
    results = la.LossAssessment(
        IdaCsv          = str(FILTERED_CSV),
        ImLevel         = IM_TARGET,
        StructuralCmp   = struct_cmp,
        CustomComponents= custom_cmp,
        ReplacementCost = REPLACEMENT_COST,
        ReplacementTime = REPLACEMENT_TIME,
        CollapseMedian  = collapse_median,
        CollapseLogStd  = collapse_logstd,
        OutputDir       = str(OUT_DIR),
    )

    # ── Step 6：输出结果 ──────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  损失评估结果汇总（FEMA P-58 方法）')
    print('=' * 65)
    print(f'  目标 IM              : {IM_TARGET} g')
    print(f'  倒塌概率             : {results["CollapseProb"]:.4f}'
          f'  ({results["CollapseProb"] * 100:.2f}%)')
    print(f'  不可修复概率         : {results["IrreparableProb"]:.4f}'
          f'  ({results["IrreparableProb"] * 100:.2f}%)')
    print(f'  平均修复费用         : {results["MeanRepairCost"]:,.0f} USD')
    print(f'  修复费用标准差       : {results["StdRepairCost"]:,.0f} USD')
    print(f'  平均修复时间         : {results["MeanRepairTime"]:,.1f} worker·day')

    agg      = results['AggLoss']
    cost_col = PelicunLossAssessment._find_col(agg, 'Cost')
    if cost_col is not None:
        q = agg[cost_col].quantile([0.16, 0.50, 0.84])
        print(f'\n  修复费用分位数:')
        print(f'    16th : {q.iloc[0]:,.0f} USD')
        print(f'    50th : {q.iloc[1]:,.0f} USD')
        print(f'    84th : {q.iloc[2]:,.0f} USD')

    print(f'\n  输出目录: {OUT_DIR}')
    print('=' * 65)
