########################################################
# 基于 Pelicun (FEMA P-58) 的建筑地震损失评估。
#
# 工作流程:
#   1. 输入 EDP（层间位移角、楼面加速度、残余位移角）
#   2. 非结构构件数量由 NormQtyPact 自动生成（需要 Windows + Excel）
#   3. 结构构件由用户以 Pelicun CMP marginals 格式提供
#   4. 可选：用户自定义构件（完整易损性 + 损失后果参数），EDP 类型须在 demand.csv 中已有
#   5. 调用 Pelicun (FEMA P-58) 执行概率性损失评估
#   6. 返回修复费用、修复时间等汇总结果
########################################################

import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from pelicun.tools.DL_calculation import run_pelicun
from pelicun import base as _pelicun_base

# EDP 短名 → Pelicun fragility CSV 所用的全名
_EDP_DEMAND_TYPE = {
    'PID': 'Peak Interstory Drift Ratio',
    'PFA': 'Peak Floor Acceleration',
    'PFV': 'Peak Floor Velocity',
    'RID': 'Residual Interstory Drift Ratio',
    'SA':  'Spectral Acceleration',
    'PGV': 'Peak Ground Velocity',
}

# EDP 短名 → fragility CSV 中 Demand-Unit 字段值
# 注：pelicun 3.9+ 将无量纲漂移角统一使用 'unitless'（老版本为 'ea'/'rad'）
_EDP_DEMAND_UNIT = {
    'PID': 'unitless',
    'PFA': 'g',
    'PFV': 'mps',
    'RID': 'unitless',
    'SA':  'g',
    'PGV': 'mps',
}


class PelicunLossAssessment:
    """
    基于 Pelicun (FEMA P-58) 的建筑地震损失评估类。

    非结构构件数量由 NormQtyPact 自动生成；结构构件由用户以
    Pelicun CMP marginals 格式提供。

    典型用法::

        # 1. 初始化（1 m² ≈ 10.764 sqft）
        la = PelicunLossAssessment(
            NumOfStories  = 6,
            FloorArea_sqft= [3240.0] * 6,   # 300 m²/层 ≈ 3230 sqft
            OccupancyType = 'OFFICE',
        )

        # 2. 定义结构构件（Pelicun CMP marginals 格式）
        struct_cmp = PelicunLossAssessment.make_struct_cmp(
            cmp_id_list = ['B.10.41.001a', 'B.10.41.001a'],
            loc_list    = ['1',            '2'           ],
            dir_list    = ['1',            '1'           ],
            qty_list    = [1.0,            1.0           ],
            unit_list   = ['ea',           'ea'          ],
        )

        # 3. 直接传入 IDA CSV（自动识别 2D/3D 格式）
        results = la.LossAssessment(
            IdaCsv        = 'IDA_results.csv',
            ImLevel       = 0.6,            # 目标 Sa (g)
            StructuralCmp = struct_cmp,
        )

        print('平均修复费用 (USD):', results['MeanRepairCost'])
        print('修复费用标准差 (USD):', results['StdRepairCost'])
        print('平均修复时间 (工人·天):', results['MeanRepairTime'])
        # results['AggLoss'] 为完整聚合损失样本 DataFrame

    OccupancyType 常用值（对应 FEMA P-58 规范化数量文件）::

        'OFFICE'       商业办公
        'APARTMENT'    住宅公寓
        'RETAIL'       零售商业
        'EDUCATION'    教育设施
        'HEALTHCARE'   医疗设施
        'WAREHOUSE'    仓储
        'HOTEL'        酒店
    """

    # ------------------------------------------------------------------ 初始化

    def __init__(
        self,
        NumOfStories: int,
        FloorArea_sqft,
        OccupancyType,
        SampleSize: int = 500,
        Seed: int = 415,
        IrreparableMedian: float = 0.01,
        IrreparableLogStd: float = 0.3,
    ):
        """
        参数
        ----
        NumOfStories : int
            楼层数。
        FloorArea_sqft : float or list[float]
            各层楼面面积（平方英尺）。
            可为标量（所有楼层相同）或长度 NumOfStories 的列表。
            单位换算：1 m² ≈ 10.764 sqft。
        OccupancyType : str or list[str]
            建筑使用类型（对应 FEMA P-58 规范化数量文件）。
            可为标量（所有楼层相同）或长度 NumOfStories 的列表。
        SampleSize : int
            蒙特卡洛模拟样本数，默认 500。
            pelicun 将从 IDA 样本拟合分布并重新采样至该数量。
        Seed : int
            随机种子，确保结果可重现，默认 415。
        IrreparableMedian : float
            不可修复小层间位移角限值中值（rad），默认 0.01（FEMA P-58 典型值）。
            当 MaxResDrift 不为 None 时展用。
        IrreparableLogStd : float
            不可修复限值对数标准差，默认 0.3。
        """
        self.NumOfStories = int(NumOfStories)
        N = self.NumOfStories

        self.FloorArea_sqft = (
            [float(FloorArea_sqft)] * N
            if isinstance(FloorArea_sqft, (int, float))
            else [float(x) for x in FloorArea_sqft]
        )
        self.OccupancyType = (
            [str(OccupancyType)] * N
            if isinstance(OccupancyType, str)
            else [str(x) for x in OccupancyType]
        )
        self.SampleSize = int(SampleSize)
        self.Seed = int(Seed)
        self.IrreparableMedian = float(IrreparableMedian)
        self.IrreparableLogStd  = float(IrreparableLogStd)

        # 评估结果（调用 LossAssessment 后填充）
        self.MeanRepairCost: float | None = None
        self.StdRepairCost: float | None = None
        self.MeanRepairTime: float | None = None
        self.AggLoss: pd.DataFrame | None = None
        self.CollapseProb: float | None = None
        self.IrreparableProb: float | None = None

    # ------------------------------------------------------------------ 辅助工具

    @staticmethod
    def make_struct_cmp(
        cmp_id_list: list,
        loc_list: list,
        dir_list: list,
        qty_list: list,
        unit_list: list = None,
        uid_list: list = None,
        family_list: list = None,
    ) -> pd.DataFrame:
        """
        创建结构构件 CMP marginals DataFrame（辅助方法）。

        参数
        ----
        cmp_id_list : list[str]
            构件 ID 列表，如 ['B.10.41.001a', 'B.10.41.001a', ...]。
            构件 ID 应与 Pelicun FEMA P-58 数据库中的 ID 对应。
        loc_list : list[str]
            各行对应的楼层编号，字符串，如 ['1', '2', '3', ...]。
        dir_list : list[str]
            方向，字符串，如 ['1', '1', ...]（'1'=X，'2'=Y）。
        qty_list : list[float]
            各行的构件数量（Theta_0）。
        unit_list : list[str], optional
            各行数量的单位，如 ['ea', 'ea', ...]。默认全部为 'ea'。
        uid_list : list[str], optional
            各行唯一标识符，默认全部为 '0'。
        family_list : list[str], optional
            分布类型，默认全部为 'deterministic'。

        返回值
        ------
        pd.DataFrame
            可直接传入 LossAssessment(StructuralCmp=...) 的 DataFrame。

        示例
        ----
        对于 6 层钢矩形框架，每层使用 'B.10.41.001a' 构件各 1 个::

            struct_cmp = PelicunLossAssessment.make_struct_cmp(
                cmp_id_list = ['B.10.41.001a'] * 6,
                loc_list    = [str(i+1) for i in range(6)],
                dir_list    = ['1'] * 6,
                qty_list    = [1.0] * 6,
                unit_list   = ['ea'] * 6,
            )
        """
        n = len(cmp_id_list)
        return pd.DataFrame(
            {
                'cmp':    cmp_id_list,
                'loc':    [str(x) for x in loc_list],
                'dir':    [str(x) for x in dir_list],
                'uid':    uid_list if uid_list is not None else ['0'] * n,
                'Theta_0': [float(x) for x in qty_list],
                'Theta_1': [np.nan] * n,
                'Family': family_list if family_list is not None else ['deterministic'] * n,
                'Blocks': [1] * n,
                'Units':  unit_list if unit_list is not None else ['ea'] * n,
            }
        )

    @staticmethod
    def make_custom_cmp(
        cmp_id_list: list,
        edp_type_list: list,
        loc_list: list,
        dir_list: list,
        qty_list: list,
        frag_theta_0: list,
        frag_theta_1: list,
        cost_theta_0: list,
        cost_theta_1: list,
        time_theta_0: list,
        time_theta_1: list,
        unit_list: list = None,
        demand_offset: list = None,
        demand_directional: list = None,
    ) -> pd.DataFrame:
        """
        创建自定义构件 DataFrame（辅助方法），支持用户完全自定义易损性和损失后果参数。

        返回的 DataFrame 可直接传入 ``LossAssessment(CustomComponents=...)``。
        自定义构件与 FEMA P-58 数据库中的标准构件一同参与评估。

        参数
        ----
        cmp_id_list : list[str]
            构件 ID 列表（自定义 ID，不能与 FEMA P-58 数据库中已有 ID 重复）。
            同一 ID 可在不同楼层重复出现。
        edp_type_list : list[str]
            各行对应的 EDP 类型，支持：'PID'、'PFA'、'PFV'、'RID'、'SA'。
            **须在 IDA 分析结果中已存在**（即 demand.csv 中已有对应列）。
            同一 cmp_id 的所有行须使用相同的 EDP 类型。
        loc_list : list[str]
            各行对应的楼层编号，字符串，如 ['1', '2', '3', ...]。
        dir_list : list[str]
            各行对应的方向，字符串，如 ['1', '1', ...] ('1'=X, '2'=Y)。
        qty_list : list[float]
            各行的构件数量。
        frag_theta_0 : list
            易损性中值。每个元素可以是：

            - **float**：单损伤状态（DS1），中值。
            - **list[float]**：多损伤状态（DS1, DS2, ...），各状态的中值。

            与 cmp_id_list 等长（同一 cmp_id 的不同楼层行须保持一致）。
        frag_theta_1 : list
            易损性对数标准差，格式与 frag_theta_0 相同。
        cost_theta_0 : list
            修复费用（USD_2011）中值，格式与 frag_theta_0 相同（各 DS 一个值）。
        cost_theta_1 : list
            修复费用对数标准差，格式与 frag_theta_0 相同（0 = 确定性）。
        time_theta_0 : list
            修复时间（worker_day）中值，格式与 frag_theta_0 相同（各 DS 一个值）。
        time_theta_1 : list
            修复时间对数标准差，格式与 frag_theta_0 相同（0 = 确定性）。
        unit_list : list[str], optional
            各行数量单位，默认全部为 'ea'。
        demand_offset : list[int], optional
            各行的 Demand-Offset（0 = 使用本层需求，1 = 使用上层需求），默认全部为 0。
        demand_directional : list[int], optional
            各行的 Demand-Directional（1 = 有方向性，0 = 无方向性），默认全部为 1。

        返回值
        ------
        pd.DataFrame
            可直接传入 ``LossAssessment(CustomComponents=...)`` 的 DataFrame。

        示例
        ----
        定义一个 6 层建筑中使用楼面速度（PFV）的自定义构件，每层 2 个，共 2 个损伤状态::

            custom_cmp = PelicunLossAssessment.make_custom_cmp(
                cmp_id_list    = ['MyEqp.001'] * 6,
                edp_type_list  = ['PFV']        * 6,
                loc_list       = [str(i+1) for i in range(6)],
                dir_list       = ['1']          * 6,
                qty_list       = [2.0]          * 6,
                frag_theta_0   = [[0.3, 0.8]]   * 6,   # DS1: 0.3 m/s, DS2: 0.8 m/s
                frag_theta_1   = [[0.4, 0.4]]   * 6,
                cost_theta_0   = [[5000, 15000]] * 6,  # DS1: 5000 USD, DS2: 15000 USD
                cost_theta_1   = [[0.4, 0.4]]   * 6,
                time_theta_0   = [[1.0, 5.0]]   * 6,   # DS1: 1 day, DS2: 5 days
                time_theta_1   = [[0.4, 0.4]]   * 6,
            )
        """
        n = len(cmp_id_list)
        # 统一将标量参数包装为列表（单 DS 情形）
        def _to_list_of_lists(vals):
            result = []
            for v in vals:
                result.append([float(v)] if not isinstance(v, (list, tuple)) else [float(x) for x in v])
            return result

        frag_t0 = _to_list_of_lists(frag_theta_0)
        frag_t1 = _to_list_of_lists(frag_theta_1)
        cost_t0 = _to_list_of_lists(cost_theta_0)
        cost_t1 = _to_list_of_lists(cost_theta_1)
        time_t0 = _to_list_of_lists(time_theta_0)
        time_t1 = _to_list_of_lists(time_theta_1)

        return pd.DataFrame({
            'cmp':                  cmp_id_list,
            'edp_type':             [str(x).upper() for x in edp_type_list],
            'loc':                  [str(x) for x in loc_list],
            'dir':                  [str(x) for x in dir_list],
            'qty':                  [float(x) for x in qty_list],
            'Units':                unit_list if unit_list is not None else ['ea'] * n,
            'demand_offset':        demand_offset if demand_offset is not None else [0] * n,
            'demand_directional':   demand_directional if demand_directional is not None else [1] * n,
            'frag_theta_0':         frag_t0,
            'frag_theta_1':         frag_t1,
            'cost_theta_0':         cost_t0,
            'cost_theta_1':         cost_t1,
            'time_theta_0':         time_t0,
            'time_theta_1':         time_t1,
        })

    # ------------------------------------------------------------------ 非结构构件生成

    def _build_nonstruct_cmp(self, tmp_dir: str) -> pd.DataFrame:
        """使用 NormQtyPact 生成非结构构件数量（Pelicun CMP marginals 格式）。

        需要 Windows 操作系统和 Microsoft Excel。
        若生成失败，调用方负责处理异常。
        """
        from normqtypact import NormQtyPact

        nqp = NormQtyPact(
            NumOfStories  = self.NumOfStories,
            FloorAreaList = self.FloorArea_sqft,
            Occupancy1Type= self.OccupancyType,
            Occupancy2Type= ['none'] * self.NumOfStories,
            Occupancy3Type= ['none'] * self.NumOfStories,
            Occupancy1Area= [1.0]   * self.NumOfStories,
            Occupancy2Area= [0.0]   * self.NumOfStories,
            Occupancy3Area= [0.0]   * self.NumOfStories,
        )
        csv_path  = str(Path(tmp_dir) / 'nonstruct_cmp.csv')
        json_path = str(Path(tmp_dir) / 'nonstruct_cmp.json')
        nqp.Output_PelicunComponentDirectory(
            json_path=json_path,
            csv_path =csv_path,
        )
        cmp_df = pd.read_csv(csv_path, index_col=0)

        # 如果有 Blocks 列，会有 Bug，不知道为什么
        cmp_df.drop(columns=['Blocks'], inplace=True)

        return cmp_df

    # ------------------------------------------------------------------ 主评估方法

    def LossAssessment(
        self,
        ImLevel: float,
        IdaCsv: 'str | Path | pd.DataFrame',
        StructuralCmp: 'pd.DataFrame | None' = None,
        CustomComponents: 'pd.DataFrame | None' = None,
        ReplacementCost: float = None,
        ReplacementTime: float = None,
        CollapseMedian: float = None,
        CollapseLogStd: float = 0.4,
        PrintLog: bool = False,
        OutputDir: 'str | Path | None' = None,
    ) -> dict:
        """
        执行基于 Pelicun / FEMA P-58 的建筑地震损失评估。

        自动识别 IDA CSV 格式：
        - 含 ``MaxDrift_X`` / ``MaxDrift_Y`` 列 → 3D IDA 结果，分别提取 X/Y 双向 EDP
          并写入 Pelicun 需求 CSV 的方向 1（X）和方向 2（Y）；
        - 否则视为标准 2D IDA 结果。

        参数
        ----
        ImLevel : float
            目标地震动强度（Sa，单位 g），用于从 IDA 结果中插值提取 EDP。
        IdaCsv : str, Path, or pd.DataFrame
            IDA 结果 CSV 文件路径或已读取的 DataFrame（由 ``MDOFModel.analysis.IDA`` 或
            ``MDOFModel.analysis.IDA_3D`` 输出）。
        StructuralCmp : pd.DataFrame, optional
            结构构件定义，由 ``make_struct_cmp()`` 生成。
            列名: cmp, loc, dir, uid, Theta_0, Theta_1, Family, Blocks, Units。
            若为 None 则仅使用非结构构件（需要 NormQtyPact）。
        CustomComponents : pd.DataFrame, optional
            用户完全自定义的构件定义，由 ``make_custom_cmp()`` 生成。
            包含易损性参数和损失后果参数，不依赖 FEMA P-58 数据库。
            **EDP 类型须在 demand.csv 中已存在**（PID、PFA、PFV、RID、SA）。
            若同时提供 StructuralCmp 和 CustomComponents，两者将合并使用。
        ReplacementCost : float, optional
            建筑替换费用（USD_2011），用于倒塌/不可修复后果模型。
            若为 None，则 pelicun 使用其内置默认值。
        ReplacementTime : float, optional
            建筑替换/倒塌修复时间（worker_day），用于倒塌/不可修复后果模型。
            若为 None，则 pelicun 使用其内置默认值（FEMA P-58 下通常为 0）。
        CollapseMedian : float, optional
            倒塌易损性中值 Sa（g）。若为 None 则不写入倒塌配置。
            写入 pelicun 配置时会自动转换为其所需的 m/s²。
        CollapseLogStd : float
            倒塌易损性对数标准差，默认 0.4。
        PrintLog : bool
            是否将 Pelicun 日志打印到终端，默认 False。
        OutputDir : str or Path, optional
            输出文件夹路径，默认为当前工作目录下的 ``pelicun_output/``。

        返回值
        ------
        dict，包含以下键值：
        'MeanRepairCost' (float) — 平均修复费用（USD）；
        'StdRepairCost' (float) — 修复费用标准差（USD）；
        'MeanRepairTime' (float or None) — 平均顺序修复时间（工人·天）；
        'CollapseProb' (float or None) — 倒塌概率（从 DL_summary.csv 的 collapse 列均值读取）；
        'IrreparableProb' (float or None) — 不可修复概率（从 DL_summary.csv 的 irreparable 列均值读取）；
        'AggLoss' (pd.DataFrame) — 完整聚合损失样本。
        """

        # ── 自动识别 2D / 3D IDA CSV 并提取 EDP ──────────────────────────
        max_drift_y     = None
        max_accel_y     = None
        max_floor_vel_y = None
        max_pgv_y       = None
        extra_edp       = {}
        extra_edp_y     = {}

        if isinstance(IdaCsv, pd.DataFrame):
            _header = IdaCsv
            _is_3d  = ('MaxDrift_X' in _header.columns and 'MaxDrift_Y' in _header.columns)
        else:
            try:
                _header = pd.read_csv(IdaCsv, nrows=0)
                _is_3d  = ('MaxDrift_X' in _header.columns and 'MaxDrift_Y' in _header.columns)
            except Exception:
                _is_3d = False

        if _is_3d:
            from ..analysis import IDA_3D as _IDA_3D
            (MaxDrift, max_drift_y,
             MaxAccel, max_accel_y,
             MaxResDrift, _res_y,
             MaxFloorVel, max_floor_vel_y,
             MaxPGV, max_pgv_y,
             extra_edp, extra_edp_y) = _IDA_3D.interp_edp_from_ida_3D(
                IdaCsv, ImLevel, self.NumOfStories
            )
            MaxResDrift = np.maximum(MaxResDrift, _res_y)
        else:
            from ..analysis import IDA as _IDA
            (MaxDrift, MaxAccel, MaxResDrift, MaxFloorVel, MaxPGV, extra_edp) = (
                _IDA.interp_edp_from_ida(IdaCsv, ImLevel, self.NumOfStories)
            )
            extra_edp_y = extra_edp  # 2D 分析两方向共用同一组 EDP

        N = self.NumOfStories
        max_drift = np.clip(np.asarray(MaxDrift, dtype=float), 1e-8, None)
        max_accel = np.clip(np.asarray(MaxAccel, dtype=float), 1e-8, None)
        max_res_drift = (
            np.clip(np.asarray(MaxResDrift, dtype=float), 1e-8, None)
            if MaxResDrift is not None else None
        )
        max_floor_vel = (
            np.clip(np.asarray(MaxFloorVel, dtype=float), 1e-8, None)
            if MaxFloorVel is not None else None
        )
        max_pgv = (
            np.clip(np.asarray(MaxPGV, dtype=float), 1e-8, None)
            if MaxPGV is not None else None
        )
        _N = self.SampleSize   # pelicun 从 IDA 样本拟合分布并重采到该数量

        # 输出文件夹：用户指定，或默认使用当前工作目录下的 pelicun_output
        work_dir = Path(OutputDir) if OutputDir is not None else Path.cwd() / 'pelicun_output'
        work_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. 生成需求 CSV ────────────────────────────────────────────────────
        demand_csv = self._build_demand_csv(
            work_dir, max_drift, max_accel, max_res_drift, max_floor_vel, ImLevel,
            max_drift_y=max_drift_y,
            max_accel_y=max_accel_y,
            max_floor_vel_y=max_floor_vel_y,
            max_pgv=max_pgv,
            max_pgv_y=max_pgv_y,
            extra_edp=extra_edp if extra_edp else None,
            extra_edp_y=extra_edp_y if extra_edp_y else None,
        )

        # ── 1b. 验证自定义构件的 EDP 类型 ─────────────────────────────────────
        if CustomComponents is not None and len(CustomComponents) > 0:
            _demand_cols = pd.read_csv(demand_csv, nrows=0).columns
            _available_edp = {
                col.split('-')[1]
                for col in _demand_cols
                if len(col.split('-')) >= 3
            }
            _bad = [
                (row['cmp'], row['edp_type'])
                for _, row in CustomComponents.iterrows()
                if str(row['edp_type']).upper() not in _available_edp
            ]
            if _bad:
                _bad_types = sorted({t for _, t in _bad})
                raise ValueError(
                    f"[LossAssessment] 自定义构件使用了 demand.csv 中不存在的 EDP 类型：{_bad_types}。\n"
                    f"  当前 demand.csv 中已有的 EDP 类型：{sorted(_available_edp)}。\n"
                    f"  请确认该 EDP 类型在 IDA 结果中已存在，或参考第二步（Task 2）"
                    f"向 IDA 结果中添加新 EDP 类型。\n"
                    f"  问题构件：{[(c, t) for c, t in _bad[:5]]}"
                )

        # ── 2. 生成构件量 CSV ─────────────────────────────────────────────────
        cmp_csv = self._build_cmp_csv(work_dir, StructuralCmp, CustomComponents)

        # ── 2b. 生成自定义构件数据库（fragility + repair consequence）─────────
        custom_fragility_db = None
        custom_repair_db    = None
        if CustomComponents is not None and len(CustomComponents) > 0:
            custom_fragility_db, custom_repair_db = self._build_custom_cmp_db(
                work_dir, CustomComponents
            )

        # ── 3. 生成 JSON 配置文件 ─────────────────────────────────────────────
        output_dir = work_dir / 'output'
        output_dir.mkdir(exist_ok=True)
        config_json = self._build_dl_config(
            work_dir, demand_csv, cmp_csv,
            N, _N, max_res_drift, ReplacementCost, ReplacementTime,
            CollapseMedian, CollapseLogStd, PrintLog,
            custom_fragility_db=custom_fragility_db,
            custom_repair_db=custom_repair_db,
        )

        # ── 4. 注入自定义 EDP 类型到 pelicun 的 demand type 映射表 ───────────
        # pelicun 的 base.EDP_to_demand_type 只内置标准类型（PID/PFA/PFV…）。
        # 对于 CustomComponents 中用户自定义的 EDP 类型（如 'STRAIN'），
        # 采用自映射 'STRAIN' → 'STRAIN'，使 pelicun 能正常解析并构建 EDP 键
        # （格式 '{TYPE}-{loc}-{dir}'），与 demand.csv 中列名 '1-{TYPE}-{loc}-1' 对应。
        if CustomComponents is not None and len(CustomComponents) > 0:
            for _raw_edp in CustomComponents['edp_type'].unique():
                _edp_up = str(_raw_edp).upper()
                # 如果不存在，注入自映射（如 'STRAIN' → 'STRAIN'）
                if _edp_up not in _EDP_DEMAND_TYPE:
                    _pelicun_base.EDP_to_demand_type[_edp_up] = _edp_up

        # ── 5. 调用 run_pelicun ───────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run_pelicun(
                config_path      = config_json,
                demand_file      = demand_csv,
                output_path      = str(output_dir),
                realizations     = _N,
                auto_script_path = None,
                custom_model_dir = None,
                output_format    = ['csv'],
                detailed_results = True,
                coupled_edp      = False,
            )

        # ── 6. 读取聚合损失结果 ───────────────────────────────────────────────
        # 不同 pelicun 版本对同一 BldgRepair 模块使用过两种输出文件名：
        # DV_repair_agg.zip 与 DV_bldg_repair_agg.zip。配置仍只使用 BldgRepair。
        #   repair_cost / repair_time-sequential / repair_time-parallel
        agg_candidates = [
            output_dir / 'DV_repair_agg.zip',
            output_dir / 'DV_bldg_repair_agg.zip',
        ]
        agg_zip = next((p for p in agg_candidates if p.exists()), None)
        if agg_zip is None:
            names = ', '.join(p.name for p in agg_candidates)
            raise FileNotFoundError(
                f'未找到 pelicun 聚合修复损失输出文件。期望文件之一: {names}; '
                f'输出目录: {output_dir}'
            )
        agg_repair = pd.read_csv(agg_zip, index_col=0, compression='zip')
        self.AggLoss = agg_repair

        # ── 7. 从 DL_summary.csv 读取倒塌概率和不可修复概率 ────────────────
        dl_summary_path = output_dir / 'DL_summary.csv'
        if dl_summary_path.exists():
            dl_summary = pd.read_csv(dl_summary_path, index_col=0)
            self.CollapseProb = (
                float(dl_summary['collapse'].mean())
                if 'collapse' in dl_summary.columns else None
            )
            self.IrreparableProb = (
                float(dl_summary['irreparable'].mean())
                if 'irreparable' in dl_summary.columns else None
            )
        else:
            self.CollapseProb = None
            self.IrreparableProb = None

        # ── 7. 汇总指标 ──────────────────────────────────────────────────────
        cost_col = self._find_col(agg_repair, 'Cost')
        time_col = self._find_col(agg_repair, 'Time')

        self.MeanRepairCost = float(agg_repair[cost_col].mean()) if cost_col is not None else 0.0
        self.StdRepairCost  = float(agg_repair[cost_col].std())  if cost_col is not None else 0.0
        self.MeanRepairTime = float(agg_repair[time_col].mean()) if time_col is not None else None

        return {
            'MeanRepairCost':  self.MeanRepairCost,
            'StdRepairCost':   self.StdRepairCost,
            'MeanRepairTime':  self.MeanRepairTime,
            'CollapseProb':    self.CollapseProb,
            'IrreparableProb': self.IrreparableProb,
            'AggLoss':         agg_repair,
        }

    # ------------------------------------------------------------------ 内部工具

    def _build_demand_csv(
        self,
        work_dir: Path,
        max_drift,
        max_accel,
        max_res_drift,
        max_floor_vel=None,
        im_level: float = None,
        max_drift_y=None,
        max_accel_y=None,
        max_floor_vel_y=None,
        max_pgv=None,
        max_pgv_y=None,
        extra_edp=None,
        extra_edp_y=None,
    ) -> str:
        """
        生成 EDP 需求 CSV 文件并写入 work_dir/demand.csv。

        CSV 格式：
          Units 行        — 各列单位（PID/RID → rad，PFA → g，PFV → mps）
          0, 1, 2, ... 行 — 各条地震记录的 EDP 样本

        max_drift       : ndarray (n_records, N)，X 方向漂移
        max_accel       : ndarray (n_records, N+1)，X 方向加速度，col 0 = 地面层
        max_res_drift   : ndarray (n_records, N) 或 (n_records,) 或 None
        max_floor_vel   : ndarray (n_records, N+1)，X 方向速度，单位 m/s；或 None
        max_drift_y     : ndarray (n_records, N)，Y 方向漂移；None 时复制 X
        max_accel_y     : ndarray (n_records, N+1)，Y 方向加速度；None 时复制 X
        max_floor_vel_y : ndarray (n_records, N+1)，Y 方向速度；None 时复制 X
        extra_edp       : dict[str, ndarray]，用户自定义 EDP（X 方向或 2D），形状 (n_records, M) 或 (n_records,)；或 None
        extra_edp_y     : dict[str, ndarray]，用户自定义 EDP（Y 方向），同上； None 时对每个 EDP 沿用 extra_edp 中的值

        列格式：TYPE-LOC-DIR（如 PID-1-1，PFA-0-1，PFV-0-1）。

        返回 CSV 文件的绝对路径字符串。
        """
        N         = self.NumOfStories
        n_records = max_drift.shape[0]

        # 若未提供 Y 方向数据，则沿用 X 方向（2D 双向相同假设）
        _drift_y   = np.asarray(max_drift_y,    dtype=float) if max_drift_y    is not None else max_drift
        _accel_y   = np.asarray(max_accel_y,    dtype=float) if max_accel_y    is not None else max_accel
        _vel_y     = (np.asarray(max_floor_vel_y, dtype=float)
                      if max_floor_vel_y is not None else max_floor_vel)

        columns: list = []
        units:   list = []

        # 方向 1（X）和方向 2（Y）漂移
        for i in range(N):
            columns.append(f'1-PID-{i + 1}-1')
            units.append('rad')
        for i in range(N):
            columns.append(f'1-PID-{i + 1}-2')
            units.append('rad')
        # 方向 1（X）和方向 2（Y）加速度
        for i in range(N + 1):
            columns.append(f'1-PFA-{i}-1')
            units.append('g')
        for i in range(N + 1):
            columns.append(f'1-PFA-{i}-2')
            units.append('g')

        # 残余位移：支持 (n, N) 或 (n,) 形状
        rid_2d = None
        if max_res_drift is not None:
            rid_arr = np.asarray(max_res_drift, dtype=float)
            rid_2d  = rid_arr if rid_arr.ndim == 2 else np.tile(rid_arr[:, None], (1, N))
            for i in range(N):
                columns.append(f'1-RID-{i + 1}-1')
                units.append('rad')

        # 楼面速度：支持 (n, N+1) 形状，单位 m/s
        pfv_x_2d = None
        pfv_y_2d = None
        if max_floor_vel is not None:
            pfv_x_2d = np.asarray(max_floor_vel, dtype=float)
            pfv_y_2d = np.asarray(_vel_y, dtype=float)
            for i in range(N + 1):
                columns.append(f'1-PFV-{i}-1')
                units.append('mps')
            for i in range(N + 1):
                columns.append(f'1-PFV-{i}-2')
                units.append('mps')

        # 谱加速度 SA：当倒塌易损性使用 SA 类型时，需要在需求 CSV 中提供对应的 SA 列（单位 g）。如果 im_level 提供，则生成一列 SA-0-1，值为 im_level。
        sa_arr = None
        if im_level is not None:
            sa_arr = np.full(n_records, float(im_level))
            columns.append('1-SA-0-1')
            units.append('g')

        # 地面峰值速度 PGV：若提供则写入 demand.csv（地面层 loc=0，单位 mps）
        pgv_x_arr = None
        pgv_y_arr = None
        if max_pgv is not None:
            pgv_x_arr = np.asarray(max_pgv, dtype=float)
            pgv_y_arr = (np.asarray(max_pgv_y, dtype=float)
                         if max_pgv_y is not None else pgv_x_arr)
            columns.append('1-PGV-0-1')
            units.append('mps')
            columns.append('1-PGV-0-2')
            units.append('mps')

        # 用户自定义额外 EDP（来自 IDA ExtraEDP 参数）
        _extra_edp_data: list = []   # list of (edp_name, x_2d, y_2d)
        if extra_edp:
            _extra_y_src = extra_edp_y if extra_edp_y else {}
            for _edp_name, _edp_arr_x in extra_edp.items():
                _x_2d = np.asarray(_edp_arr_x, dtype=float)
                _y_src = _extra_y_src.get(_edp_name, _edp_arr_x)
                _y_2d = np.asarray(_y_src, dtype=float)
                # 统一变为 2D：(n_records, M)
                if _x_2d.ndim == 1:
                    _x_2d = _x_2d[:, np.newaxis]
                if _y_2d.ndim == 1:
                    _y_2d = _y_2d[:, np.newaxis]
                _n_locs = _x_2d.shape[1]
                _unit = _EDP_DEMAND_UNIT.get(_edp_name.upper(), 'unitless')
                for _i in range(_n_locs):
                    columns.append(f'1-{_edp_name}-{_i + 1}-1')
                    units.append(_unit)
                for _i in range(_n_locs):
                    columns.append(f'1-{_edp_name}-{_i + 1}-2')
                    units.append(_unit)
                _extra_edp_data.append((_edp_name, _x_2d, _y_2d))

        # 构建数据行（每条地震记录一行）
        data_rows = []
        for r in range(n_records):
            # 方向 1（X）+ 方向 2（Y）的漂移和加速度
            row = (
                list(max_drift[r]) + list(_drift_y[r])
                + list(max_accel[r]) + list(_accel_y[r])
            )
            if rid_2d is not None:
                row += list(rid_2d[r])
            if pfv_x_2d is not None:
                row += list(pfv_x_2d[r]) + list(pfv_y_2d[r])
            if sa_arr is not None:
                row += [sa_arr[r]]
            if pgv_x_arr is not None:
                row += [pgv_x_arr[r], pgv_y_arr[r]]
            for _, _x_2d, _y_2d in _extra_edp_data:
                row += list(_x_2d[r]) + list(_y_2d[r])
            data_rows.append(row)

        demand_df = pd.DataFrame(
            [units] + data_rows,
            index=pd.Index(['Units'] + [str(i) for i in range(n_records)], name='ID'),
            columns=columns,
        )
        demand_csv = str(work_dir / 'demand.csv')
        demand_df.to_csv(demand_csv)
        return demand_csv

    def _build_cmp_csv(
        self,
        work_dir: Path,
        StructuralCmp: 'pd.DataFrame | None',
        CustomComponents: 'pd.DataFrame | None' = None,
    ) -> str:
        """
        生成构件数量 CSV 文件（CMP_QNT 格式）并写入 work_dir/CMP_QNT.csv。

        CMP_QNT 格式：index = cmp_id，
        columns = [Units, Location, Direction, Theta_0, Family]

        来源优先级：
        1. NormQtyPact 生成的非结构构件（需要 Windows + Excel）；
        2. make_struct_cmp() 格式的结构构件；
        3. make_custom_cmp() 格式的自定义构件（量定义部分）。
        若三者均不可用，写入一个占位行以避免 pelicun 因空资产模型报错。

        返回 CSV 文件的绝对路径字符串。
        """
        cmp_parts = []

        # 非结构构件（NormQtyPact，需要 Windows + Excel）
        try:
            nonstruct_df = self._build_nonstruct_cmp(str(work_dir))
            if len(nonstruct_df) > 0:
                cmp_parts.append(nonstruct_df)
        except Exception as exc:
            warnings.warn(
                f'[PelicunLossAssessment] NormQtyPact 生成非结构构件失败，将跳过。\n'
                f'  原因: {exc}\n'
                f'  提示: NormQtyPact 仅支持 Windows + Microsoft Excel。',
                stacklevel=3,
            )

        # 结构构件：make_struct_cmp 格式 → CMP_QNT 格式
        if StructuralCmp is not None and len(StructuralCmp) > 0:
            cmp_parts.append(pd.DataFrame({
                'Units':     StructuralCmp['Units'].values,
                'Location':  StructuralCmp['loc'].values,
                'Direction': StructuralCmp['dir'].values,
                'Theta_0':   StructuralCmp['Theta_0'].values,
                'Family': [
                    'N/A' if pd.isna(x) or str(x) in ('deterministic', 'nan')
                    else str(x)
                    for x in StructuralCmp['Family'].values
                ],
            }, index=pd.Index(StructuralCmp['cmp'].values)))

        # 自定义构件：make_custom_cmp 格式 → CMP_QNT 格式（仅量定义部分）
        if CustomComponents is not None and len(CustomComponents) > 0:
            cmp_parts.append(pd.DataFrame({
                'Units':     CustomComponents['Units'].values,
                'Location':  CustomComponents['loc'].values,
                'Direction': CustomComponents['dir'].values,
                'Theta_0':   CustomComponents['qty'].values,
                'Family':    ['N/A'] * len(CustomComponents),
            }, index=pd.Index(CustomComponents['cmp'].values)))

        cmp_df = pd.concat(cmp_parts) if cmp_parts else pd.DataFrame(
            {'Units': ['ea'], 'Location': ['1'], 'Direction': ['1'],
             'Theta_0': [0.0], 'Family': ['N/A']},
            index=pd.Index(['placeholder']),
        )
        cmp_csv = str(work_dir / 'CMP_QNT.csv')
        cmp_df.to_csv(cmp_csv)
        return cmp_csv

    def _build_custom_cmp_db(
        self,
        work_dir: Path,
        custom_cmp_df: pd.DataFrame,
    ) -> 'tuple[str, str]':
        """
        将 make_custom_cmp() 生成的自定义构件 DataFrame 写入 Pelicun 格式的
        fragility 和 repair consequence CSV 文件（仅包含自定义构件行）。

        pelicun 3.9+ 支持叠加数据库：通过同时设置 ComponentDatabase='FEMA P-58'
        和 ComponentDatabasePath 指向本文件，pelicun 会将两者合并加载。
        因此无需手动读取或合并 FEMA P-58 基础数据库。

        返回 (custom_fragility_csv_path, custom_consequence_repair_csv_path)。
        """
        # ── 对每个唯一 cmp_id，仅取第一行定义的易损性/后果参数（要求一致性）──
        seen_ids = {}
        for _, row in custom_cmp_df.iterrows():
            cid = row['cmp']
            if cid not in seen_ids:
                seen_ids[cid] = row

        # ── 构建自定义 fragility CSV ──────────────────────────────────────
        frag_max_ds = max(len(row['frag_theta_0']) for row in seen_ids.values())
        frag_cols = ['Incomplete', 'Demand-Type', 'Demand-Unit', 'Demand-Offset', 'Demand-Directional']
        for ds_i in range(1, frag_max_ds + 1):
            frag_cols += [f'LS{ds_i}-Family', f'LS{ds_i}-Theta_0', f'LS{ds_i}-Theta_1', f'LS{ds_i}-DamageStateWeights']

        frag_rows = {}
        for cid, row in seen_ids.items():
            edp = row['edp_type'].upper()
            demand_type = _EDP_DEMAND_TYPE.get(edp, edp)
            demand_unit = _EDP_DEMAND_UNIT.get(edp, 'unitless')
            frag_row: dict = {
                'Incomplete':         0,
                'Demand-Type':        demand_type,
                'Demand-Unit':        demand_unit,
                'Demand-Offset':      int(row['demand_offset']),
                'Demand-Directional': int(row['demand_directional']),
            }
            thetas_0 = row['frag_theta_0']
            thetas_1 = row['frag_theta_1']
            for ds_i, (t0, t1) in enumerate(zip(thetas_0, thetas_1), start=1):
                frag_row[f'LS{ds_i}-Family']  = 'lognormal'
                frag_row[f'LS{ds_i}-Theta_0'] = float(t0)
                frag_row[f'LS{ds_i}-Theta_1'] = float(t1)
            frag_rows[cid] = frag_row

        custom_frag = pd.DataFrame(frag_rows).T
        custom_frag.index.name = 'ID'
        for col in frag_cols:
            if col not in custom_frag.columns:
                custom_frag[col] = np.nan
        custom_frag = custom_frag[frag_cols]

        custom_frag_path = str(work_dir / 'custom_fragility.csv')
        custom_frag.to_csv(custom_frag_path)

        # ── 构建自定义 repair consequence CSV ─────────────────────────────
        repair_max_ds = max(len(row['cost_theta_0']) for row in seen_ids.values())
        repair_cols = ['Incomplete', 'Quantity-Unit', 'DV-Unit']
        for ds_i in range(1, repair_max_ds + 1):
            repair_cols += [f'DS{ds_i}-Family', f'DS{ds_i}-Theta_0', f'DS{ds_i}-Theta_1', f'DS{ds_i}-LongLeadTime']

        repair_rows = {}
        for cid, row in seen_ids.items():
            cost_t0 = row['cost_theta_0']
            cost_t1 = row['cost_theta_1']
            time_t0 = row['time_theta_0']
            time_t1 = row['time_theta_1']
            qty_unit = str(row.get('Units', 'ea')).upper()

            for dv, t0_list, t1_list, dv_unit in [
                ('Cost', cost_t0, cost_t1, 'USD_2011'),
                ('Time', time_t0, time_t1, 'worker_day'),
            ]:
                r: dict = {
                    'Incomplete':    0,
                    'Quantity-Unit': f'1 {qty_unit}',
                    'DV-Unit':       dv_unit,
                }
                for ds_i, (t0, t1) in enumerate(zip(t0_list, t1_list), start=1):
                    r[f'DS{ds_i}-Family']  = 'lognormal' if float(t1) > 0 else 'deterministic'
                    r[f'DS{ds_i}-Theta_0'] = float(t0)
                    if float(t1) > 0:
                        r[f'DS{ds_i}-Theta_1'] = float(t1)
                    if dv == 'Time':
                        r[f'DS{ds_i}-LongLeadTime'] = 0
                repair_rows[f'{cid}-{dv}'] = r

        custom_repair = pd.DataFrame(repair_rows).T
        custom_repair.index.name = 'ID'
        for col in repair_cols:
            if col not in custom_repair.columns:
                custom_repair[col] = np.nan
        custom_repair = custom_repair[repair_cols]

        custom_repair_path = str(work_dir / 'custom_consequence_repair.csv')
        custom_repair.to_csv(custom_repair_path)

        return custom_frag_path, custom_repair_path

    def _build_dl_config(
        self,
        work_dir: Path,
        demand_csv: str,
        cmp_csv: str,
        num_stories: int,
        sample_size: int,
        max_res_drift,
        replacement_cost: 'float | None',
        replacement_time: 'float | None',
        collapse_median: 'float | None',
        collapse_logstd: float,
        print_log: bool,
        custom_fragility_db: 'str | None' = None,
        custom_repair_db: 'str | None' = None,
    ) -> str:
        """
        生成 Pelicun JSON 配置文件并写入 work_dir/DL_config.json。

        配置格式遵循 pelicun input_schema.json（GeneralInformation + DL 结构）；
        所有文件路径均使用绝对路径字符串。

        - 当 max_res_drift 不为 None 时，启用 ``IrreparableDamage``，参数取自
          ``self.IrreparableMedian`` 与 ``self.IrreparableLogStd``。
        - 当 collapse_median 不为 None 时，写入 ``CollapseFragility``（以 SA 为
          需求类型）。本封装的 collapse_median 输入单位为 g；pelicun 在
          ``length=m`` 配置下要求 SA 容量使用 m/s^2，因此写入配置前需要转换。
        - 当 custom_fragility_db / custom_repair_db 不为 None 时，在 'FEMA P-58' 基础上额外指定 ComponentDatabasePath / ConsequenceDatabasePath，
          pelicun 3.9+ 会将两者叠加加载（无需手动合并）。

        返回 JSON 文件的绝对路径字符串。
        """
        demands_cfg: dict = {
            'DemandFilePath': demand_csv,
            # CoupledDemands 不设置（取默认 False）：pelicun 拟合分布并重采到 SampleSize
        }

        damage_cfg: dict = {
            'DamageProcess': 'FEMA P-58',
        }
        if max_res_drift is not None:
            damage_cfg['IrreparableDamage'] = {
                'DriftCapacityMedian': self.IrreparableMedian,
                'DriftCapacityLogStd': self.IrreparableLogStd,
            }
        if collapse_median is not None:
            # pelicun 的 CollapseFragility 不支持在 CapacityMedian 字段中直接写
            # "1.5 g" 这样的带单位字符串；它会根据 Demand Unit 转换纯数值。
            # 这里 GeneralInformation.units.length 固定为 m，因此 SA 容量单位为 m/s^2。
            g_to_mps2 = 9.80665
            damage_cfg['CollapseFragility'] = {
                'DemandType':           'SA',
                'CapacityDistribution': 'lognormal',
                'CapacityMedian':       float(collapse_median) * g_to_mps2,
                'Theta_1':              float(collapse_logstd),
            }

        # pelicun 3.9+ 支持叠加：ComponentDatabase='FEMA P-58' 同时指定
        # ComponentDatabasePath，两者会合并加载，无需手动合并 CSV
        asset_db_cfg: dict = {'ComponentDatabase': 'FEMA P-58'}
        if custom_fragility_db is not None:
            asset_db_cfg['ComponentDatabasePath'] = custom_fragility_db

        repair_db_cfg: dict = {'ConsequenceDatabase': 'FEMA P-58'}
        if custom_repair_db is not None:
            repair_db_cfg['ConsequenceDatabasePath'] = custom_repair_db

        repair_cfg: dict = {
            **repair_db_cfg,
            'MapApproach':       'Automatic',
            'DecisionVariables': {'Cost': True, 'Time': True},
        }

        dl_config: dict = {
            'GeneralInformation': {
                'units': {'length': 'm'},
            },
            'DL': {
                'Demands': demands_cfg,
                'Asset': {
                    'ComponentAssignmentFile': cmp_csv,
                    **asset_db_cfg,
                    'NumberOfStories':         str(num_stories),
                    'OccupancyType':           self.OccupancyType[0],
                },
                'Damage': damage_cfg,
                'Losses': {
                    'BldgRepair': repair_cfg,
                },
                'Options': {
                    'Seed':     self.Seed,
                    'PrintLog': print_log,
                    'LogFile':  str(work_dir / 'pelicun_log.txt'),
                    'Sampling': {'SampleSize': sample_size},
                },
                'Outputs': {
                    'Demand': {
                        'Sample': True,
                        'Statistics': True,
                    },
                    'Asset': {
                        'Sample': True,
                        'Statistics': True,
                    },
                    'Damage': {
                        'Sample': True,
                        'Statistics': True,
                        'GroupedSample': True,
                        'GroupedStatistics': True,
                    },
                    'Loss': {
                        'BldgRepair': {
                            'Sample':              True,
                            'Statistics':          True,
                            'GroupedSample':       True,
                            'GroupedStatistics':   True,
                            'AggregateSample':     True,
                            'AggregateStatistics': True,
                        },
                    },
                    'Format': {'CSV': True, 'JSON': False},
                },
            },
        }

        if replacement_cost is not None:
            dl_config['DL']['Losses']['BldgRepair']['ReplacementCost'] = {
                'Median': float(replacement_cost),
                'Unit':   'USD_2011',
            }
        if replacement_time is not None:
            dl_config['DL']['Losses']['BldgRepair']['ReplacementTime'] = {
                'Median': float(replacement_time),
                'Unit':   'worker_day',
            }

        config_json = str(work_dir / 'DL_config.json')
        with open(config_json, 'w', encoding='utf-8') as fp:
            json.dump(dl_config, fp, indent=2)
        return config_json

    @staticmethod
    def _find_col(df: pd.DataFrame, key: str):
        """
        在 DataFrame 列中查找包含 key 的列。

        支持两种格式：
        - MultiIndex：pelicun 内部 API 直接返回的 (dv, category) 格式
        - 简单字符串：从 DV_repair_agg.zip / DV_bldg_repair_agg.zip 读取的 'repair_cost',
          'repair_time-sequential', 'repair_time-parallel' 等格式
        """
        if df is None or len(df.columns) == 0:
            return None
        key_lower = key.lower()
        if isinstance(df.columns, pd.MultiIndex):
            # 优先找 (key, 'total')，否则找第一个 (key, ...)
            candidates = [c for c in df.columns
                          if c[0] == key and 'total' in str(c[1]).lower()]
            if not candidates:
                candidates = [c for c in df.columns if c[0] == key]
        else:
            # 简单字符串列名（从 ZIP 文件读取）：
            #   Cost → repair_cost
            #   Time → repair_time-sequential（优先）或 repair_time-parallel
            exact = [c for c in df.columns
                     if str(c).lower() == f'repair_{key_lower}']
            if exact:
                return exact[0]
            # 时间 DV：优先 sequential（总工期），不要 parallel
            sequential = [c for c in df.columns
                          if key_lower in str(c).lower()
                          and 'sequential' in str(c).lower()]
            if sequential:
                return sequential[0]
            candidates = [c for c in df.columns
                          if key_lower in str(c).lower()
                          and 'parallel' not in str(c).lower()]
            if not candidates:
                candidates = [c for c in df.columns
                              if key_lower in str(c).lower()]
        return candidates[0] if candidates else None
