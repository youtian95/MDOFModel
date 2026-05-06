########################################################
# 基于 Pelicun (FEMA P-58) 的建筑地震损失评估。
#
# 工作流程:
#   1. 输入 EDP（层间位移角、楼面加速度、残余位移角）
#   2. 非结构构件数量由 NormQtyPact 自动生成（需要 Windows + Excel）
#   3. 结构构件由用户以 Pelicun CMP marginals 格式提供
#   4. 调用 Pelicun (FEMA P-58) 执行概率性损失评估
#   5. 返回修复费用、修复时间等汇总结果
########################################################

import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


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

        # 3. 运行评估
        results = la.LossAssessment(
            MaxDrift      = max_drift_list,     # rad，长度 = NumOfStories
            MaxAccel      = max_accel_list,     # g，长度 = NumOfStories+1（含地面）
            ResDrift      = float(res_drift),   # rad，最大残余层间位移角
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

    # ------------------------------------------------------------------ IDA 结果插值

    def interp_edp_from_ida(
        self,
        ida_csv: 'str | Path',
        im_target: float,
    ):
        """
        从 IDA 结果 CSV 中提取目标 IM 水平处各条地震波的原始 EDP 样本。

        兼容入口。实际计算由 ``MDOFModel.analysis.IDA.interp_edp_from_ida``
        完成，并对每条地震波在相邻 IDA 强度水平之间进行线性插值。

        IDA 结果 CSV 应包含以下列（由 MDOFModel.analysis.IDA 输出）：
          - ``IM``          : 地震动强度水平（Sa，g）
          - ``Iffinish``    : 是否完成计算的布尔标志
          - ``MaxDrift``    : 各层最大层间位移角（rad），字符串/数组格式
          - ``MaxAbsAccel`` : 各楼面最大绝对加速度（mm/s²），字符串/数组格式
          - ``ResDrift``    : 最大残余层间位移角（rad），标量或数组

        参数
        ----
        ida_csv : str or Path
            IDA 结果 CSV 文件路径（通常为 ``IDA_results.csv``）。
        im_target : float
            目标地震动强度（Sa，单位 g）；在相邻 IDA 强度水平之间线性插值。

        返回值
        ------
        drift_mat : np.ndarray, shape (n_records, NumOfStories)
            各条记录各层 IDR 样本，单位 rad。
        accel_mat : np.ndarray, shape (n_records, NumOfStories+1)
            各条记录各楼面 PFA 样本，单位 g。
            第 0 列为地面层 PGA（以 0.4×im_target 近似）。
        res_arr : np.ndarray, shape (n_records,)
            各条记录全楼最大 RID 样本，单位 rad。
        vel_mat : np.ndarray, shape (n_records, NumOfStories+1)
            各条记录各楼面 PFV 样本，单位 m/s。
            第 0 列为地面层，第 i 列为第 i 层楼面。
            若 IDA CSV 中无速度列，则以 10.0 m/s 填充（保守估计）。
        """
        warnings.warn(
            "PelicunLossAssessment.interp_edp_from_ida is kept for compatibility. "
            "Use MDOFModel.analysis.IDA.interp_edp_from_ida instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..analysis import IDA

        return IDA.interp_edp_from_ida(ida_csv, im_target, self.NumOfStories)

    # ------------------------------------------------------------------ 主评估方法

    def LossAssessment(
        self,
        MaxDrift,
        MaxAccel,
        StructuralCmp: 'pd.DataFrame | None' = None,
        ReplacementCost: float = None,
        CollapseMedian: float = None,
        CollapseLogStd: float = 0.4,
        PrintLog: bool = False,
        MaxResDrift=None,
        MaxFloorVel=None,
        ImLevel: float = None,
        OutputDir: 'str | Path | None' = None,
    ) -> dict:
        """
        执行基于 Pelicun / FEMA P-58 的建筑地震损失评估。

        内部调用 ``pelicun.tools.DL_calculation.run_pelicun``，以 JSON 配置文件
        + EDP 需求 CSV 作为输入，规避底层 API 变动带来的兼容风险。
        中间/输出文件写入 ``OutputDir``（默认为当前工作目录下的 ``pelicun_output/``），
        评估完成后文件保留，可直接检查。

        参数
        ----
        MaxDrift : array-like, shape (n_records, NumOfStories)
            各条地震记录各层最大层间位移角（IDR），单位 rad。
            可由 ``MDOFModel.analysis.IDA.interp_edp_from_ida`` 直接获取。
        MaxAccel : array-like, shape (n_records, NumOfStories+1)
            各条地震记录各楼面最大绝对加速度，单位 g。
            第 0 列为地面层加速度，第 i 列为第 i 层楼面加速度。
            可由 ``MDOFModel.analysis.IDA.interp_edp_from_ida`` 直接获取。
        StructuralCmp : pd.DataFrame, optional
            结构构件定义，由 ``make_struct_cmp()`` 生成。
            列名: cmp, loc, dir, uid, Theta_0, Theta_1, Family, Blocks, Units。
            若为 None 则仅使用非结构构件（需要 NormQtyPact）。
        ReplacementCost : float, optional
            建筑替换费用（USD_2011），用于倒塌/不可修复后果模型。
            若为 None，则 pelicun 使用其内置默认值。
        PrintLog : bool
            是否将 Pelicun 运行日志打印到终端，默认 False。
        CollapseMedian : float, optional
            倒塌易损性中值 Sa（g）。若为 None 则不写入倒塌配置。
        CollapseLogStd : float
            倒塌易损性对数标准差，默认 0.4。
        MaxResDrift : array-like, shape (n_records, NumOfStories) or (n_records,), optional
            各条地震记录残余层间位移角（RID），单位 rad。
            若为一维数组，各层共用同一最大值。
            若提供，启用 FEMA P-58 不可修复损失评估（``IrreparableDamage``）。
            若为 None，则跳过残余位移相关计算。
        OutputDir : str or Path, optional
            输出文件夹路径。若为 None，则默认使用当前工作目录下的 ``pelicun_output/``。
            文件夹在评估完成后保留，可用于调试检查。

        返回值
        ------
        dict
            ``'MeanRepairCost'`` : float — 平均修复费用（USD）。
            ``'StdRepairCost'``  : float — 修复费用标准差（USD）。
            ``'MeanRepairTime'`` : float or None — 平均顺序修复时间（工人·天）。
            ``'AggLoss'``        : pd.DataFrame — 完整聚合损失样本。
        """
        try:
            from pelicun.tools.DL_calculation import run_pelicun
        except ImportError as exc:
            raise ImportError(
                "找不到 pelicun 包。请运行: pip install pelicun"
            ) from exc

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
        _N = self.SampleSize   # pelicun 从 IDA 样本拟合分布并重采到该数量

        # 输出文件夹：用户指定，或默认使用当前工作目录下的 pelicun_output
        work_dir = Path(OutputDir) if OutputDir is not None else Path.cwd() / 'pelicun_output'
        work_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. 生成需求 CSV ────────────────────────────────────────────────────
        demand_csv = self._build_demand_csv(work_dir, max_drift, max_accel, max_res_drift, max_floor_vel, ImLevel)

        # ── 2. 生成构件量 CSV ─────────────────────────────────────────────────
        cmp_csv = self._build_cmp_csv(work_dir, StructuralCmp)

        # ── 3. 生成 JSON 配置文件 ─────────────────────────────────────────────
        output_dir = work_dir / 'output'
        output_dir.mkdir(exist_ok=True)
        config_json = self._build_dl_config(
            work_dir, demand_csv, cmp_csv,
            N, _N, max_res_drift, ReplacementCost,
            CollapseMedian, CollapseLogStd, PrintLog,
        )

        # ── 4. 调用 run_pelicun ───────────────────────────────────────────────
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

        # ── 5. 读取聚合损失结果 ───────────────────────────────────────────────
        # DV_repair_agg.zip 列名格式：
        #   repair_cost / repair_time-sequential / repair_time-parallel
        agg_zip = output_dir / 'DV_repair_agg.zip'
        agg_repair = pd.read_csv(agg_zip, index_col=0, compression='zip')
        self.AggLoss = agg_repair

        # ── 6. 汇总指标 ──────────────────────────────────────────────────────
        cost_col = self._find_col(agg_repair, 'Cost')
        time_col = self._find_col(agg_repair, 'Time')

        self.MeanRepairCost = float(agg_repair[cost_col].mean()) if cost_col is not None else 0.0
        self.StdRepairCost  = float(agg_repair[cost_col].std())  if cost_col is not None else 0.0
        self.MeanRepairTime = float(agg_repair[time_col].mean()) if time_col is not None else None

        return {
            'MeanRepairCost': self.MeanRepairCost,
            'StdRepairCost':  self.StdRepairCost,
            'MeanRepairTime': self.MeanRepairTime,
            'AggLoss':        agg_repair,
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
    ) -> str:
        """
        生成 EDP 需求 CSV 文件并写入 work_dir/demand.csv。

        CSV 格式：
          Units 行        — 各列单位（PID/RID → rad，PFA → g，PFV → mps）
          0, 1, 2, ... 行 — 各条地震记录的 EDP 样本

        max_drift       : ndarray (n_records, N)
        max_accel       : ndarray (n_records, N+1)，col 0 = 地面加速度
        max_res_drift   : ndarray (n_records, N) 或 (n_records,) 或 None
        max_floor_vel   : ndarray (n_records, N+1)，col 0 = 地面层速度，单位 m/s；或 None

        列格式：TYPE-LOC-DIR（如 PID-1-1，PFA-0-1，PFV-0-1）。

        返回 CSV 文件的绝对路径字符串。
        """
        N         = self.NumOfStories
        n_records = max_drift.shape[0]

        columns: list = []
        units:   list = []

        # 同时写入方向 1 和方向 2（非结构构件通常为双向，pelicun 需要两个方向均存在）
        for i in range(N):
            columns.append(f'1-PID-{i + 1}-1')
            units.append('rad')
        for i in range(N):
            columns.append(f'1-PID-{i + 1}-2')
            units.append('rad')
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
        pfv_2d = None
        if max_floor_vel is not None:
            pfv_2d = np.asarray(max_floor_vel, dtype=float)
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

        # 构建数据行（每条地震记录一行）
        data_rows = []
        for r in range(n_records):
            # 方向 1 + 方向 2（相同值，2D 模型按双向相同假设）
            row = list(max_drift[r]) + list(max_drift[r]) + list(max_accel[r]) + list(max_accel[r])
            if rid_2d is not None:
                row += list(rid_2d[r])
            if pfv_2d is not None:
                row += list(pfv_2d[r]) + list(pfv_2d[r])
            if sa_arr is not None:
                row += [sa_arr[r]]
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
    ) -> str:
        """
        生成构件数量 CSV 文件（CMP_QNT 格式）并写入 work_dir/CMP_QNT.csv。

        CMP_QNT 格式：index = cmp_id，
        columns = [Units, Location, Direction, Theta_0, Family]

        来源优先级：
        1. NormQtyPact 生成的非结构构件（需要 Windows + Excel）；
        2. make_struct_cmp() 格式的结构构件。
        若两者均不可用，写入一个占位行以避免 pelicun 因空资产模型报错。

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

        cmp_df = pd.concat(cmp_parts) if cmp_parts else pd.DataFrame(
            {'Units': ['ea'], 'Location': ['1'], 'Direction': ['1'],
             'Theta_0': [0.0], 'Family': ['N/A']},
            index=pd.Index(['placeholder']),
        )
        cmp_csv = str(work_dir / 'CMP_QNT.csv')
        cmp_df.to_csv(cmp_csv)
        return cmp_csv

    def _build_dl_config(
        self,
        work_dir: Path,
        demand_csv: str,
        cmp_csv: str,
        num_stories: int,
        sample_size: int,
        max_res_drift,
        replacement_cost: 'float | None',
        collapse_median: 'float | None',
        collapse_logstd: float,
        print_log: bool,
    ) -> str:
        """
        生成 Pelicun JSON 配置文件并写入 work_dir/DL_config.json。

        配置格式遵循 pelicun input_schema.json（GeneralInformation + DL 结构）；
        所有文件路径均使用绝对路径字符串。

        - 当 max_res_drift 不为 None 时，启用 ``IrreparableDamage``，参数取自
          ``self.IrreparableMedian`` 与 ``self.IrreparableLogStd``。
        - 当 collapse_median 不为 None 时，写入 ``CollapseFragility``（以 SA 为
          需求类型）。

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
            damage_cfg['CollapseFragility'] = {
                'DemandType':      'SA',
                'CapacityMedian':  float(collapse_median),
                'Theta_1':         float(collapse_logstd),
            }

        dl_config: dict = {
            'GeneralInformation': {
                'units': {'length': 'ft'},
            },
            'DL': {
                'Demands': demands_cfg,
                'Asset': {
                    'ComponentAssignmentFile': cmp_csv,
                    'ComponentDatabase':       'FEMA P-58',
                    'NumberOfStories':         str(num_stories),
                    'OccupancyType':           self.OccupancyType[0],
                },
                'Damage': damage_cfg,
                'Losses': {
                    'Repair': {
                        'ConsequenceDatabase': 'FEMA P-58',
                        'MapApproach':         'Automatic',
                        'DecisionVariables':   {'Cost': True, 'Time': True},
                    },
                },
                'Options': {
                    'Seed':     self.Seed,
                    'PrintLog': print_log,
                    'LogFile':  str(work_dir / 'pelicun_log.txt'),
                    'Sampling': {'SampleSize': sample_size},
                },
                # 仅输出聚合损失样本，最小化写盘量
                'Outputs': {
                    'Loss': {
                        'Repair': {
                            'AggregateSample':     True,
                            'AggregateStatistics': True,
                        },
                    },
                    'Format': {'CSV': True, 'JSON': False},
                },
            },
        }

        if replacement_cost is not None:
            dl_config['DL']['Losses']['Repair']['ReplacementCost'] = {
                'Median': float(replacement_cost),
                'Unit':   'USD_2011',
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
        - 简单字符串：从 DV_repair_agg.zip 读取的 'repair_cost',
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
