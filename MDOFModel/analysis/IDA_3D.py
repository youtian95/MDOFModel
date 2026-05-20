########################################################
# 基于 FEMA P-695 双向地震动记录对（X 分量 + Y 分量）的 3D IDA 分析模块。
#
# 分析策略：
#   对于平面结构（GeneralModelWrapper 单向模型），将同一结构分别在
#   X 和 Y 激励下独立运行，等效模拟双向地震响应（适用于平面对称结构）。
#
# 强度指标（IM）：
#   取两分量在 T₁ 处谱加速度的几何均值：Sa_gm = sqrt(Sa_X × Sa_Y)
#   两分量使用统一的缩放系数：SF = IM_target / Sa_gm
#
# 输出 CSV 列格式：
#   IM, EQRecord_X, EQRecord_Y,
#   MaxDrift_X, MaxDrift_Y,          (各层列表，字符串格式)
#   MaxAbsAccel_X, MaxAbsAccel_Y,    (各层列表，mm/s²)
#   MaxAbsVel_X, MaxAbsVel_Y,        (各层列表，mm/s)
#   ResDrift_X, ResDrift_Y,          (标量，rad)
#   Iffinish, Iffinish_X, Iffinish_Y,
#   tCurrent_X, tCurrent_Y, TotalTime
#
# 依赖库:
#   openseespy, pandas, numpy, eqsig, tqdm
########################################################

from collections import Counter
import copy
import math
import multiprocessing as mp
import threading
from pathlib import Path
from typing import Union

import eqsig.single
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import tempfile

from . import ReadRecord as _ReadRecord_module
from .IDA import IDAModelProtocol, _parse_ida_array, _interp_ida_value
# IDA_3D CSV 中固定存在的非自定义 EDP 列名集合
_IDA3D_STANDARD_COLS = frozenset({
    'IM', 'EQRecord_X', 'EQRecord_Y',
    'MaxDrift_X', 'MaxDrift_Y', 'MaxAbsAccel_X', 'MaxAbsAccel_Y', 'MaxAbsVel_X', 'MaxAbsVel_Y', 'PGV_X', 'PGV_Y', 'ResDrift_X', 'ResDrift_Y', 'Iffinish', 'Iffinish_X', 'Iffinish_Y', 'tCurrent_X', 'tCurrent_Y', 'TotalTime', '_pair',
})

# ─────────────────────────────────────────────────────────────── 工具函数


def _compute_sa(record_file: str, period: float) -> float:
    """计算单条地震动记录在给定周期处的谱加速度（g）。"""
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w', delete=False) as _f:
        _tmp_path = _f.name
    try:
        dt, _ = _ReadRecord_module.ReadRecord(record_file, _tmp_path)
        accel = np.array(open(_tmp_path).read().split(), dtype=float)
    finally:
        Path(_tmp_path).unlink(missing_ok=True)
    rec = eqsig.single.AccSignal(accel * 9.8, dt)
    rec.generate_response_spectrum(response_times=np.array([period]))
    return float(rec.s_a[0] / 9.8)


def _compute_pgv(record_file: str) -> float:
    """计算单条地震动记录的地面峰值速度（PGV，m/s），对加速度时程进行梯形积分。"""
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w', delete=False) as _f:
        _tmp_path = _f.name
    try:
        dt, _ = _ReadRecord_module.ReadRecord(record_file, _tmp_path)
        accel_g = np.array(open(_tmp_path).read().split(), dtype=float)
    finally:
        Path(_tmp_path).unlink(missing_ok=True)
    accel_ms2 = accel_g * 9.8
    vel = np.zeros(len(accel_ms2))
    if len(accel_ms2) > 1:
        vel[1:] = np.cumsum((accel_ms2[:-1] + accel_ms2[1:]) * 0.5 * dt)
    return float(np.max(np.abs(vel)))


def _sa_geomean(record_x: str, record_y: str, period: float) -> tuple:
    """计算记录对的单分量 Sa 及几何均值 Sa（g）。

    Returns
    -------
    tuple
        (Sa_X, Sa_Y, Sa_gm) 均为 g 单位。
    """
    sa_x = _compute_sa(record_x, period)
    sa_y = _compute_sa(record_y, period)
    sa_gm = math.sqrt(sa_x * sa_y) if sa_x > 0 and sa_y > 0 else max(sa_x, sa_y, 1e-9)
    return sa_x, sa_y, sa_gm


def _to_float(val) -> float:
    """将标量或列表形式的残余位移转换为单一浮点数。"""
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val, dtype=float)
        return float(arr.max()) if arr.size > 0 else 0.0
    return float(val)


# ─────────────────────────────────────────────────────────── 单记录对分析


def _IDA3D_1record_pair(
    FEModel: IDAModelProtocol,
    IM_list: list,
    EQRecordX: str,
    EQRecordY: str,
    period: float,
    DeltaT='AsInRecord',
    _status_queue=None,
    ExtraEDP=None,
) -> pd.DataFrame:
    """对一个地震动记录对（X 分量 + Y 分量）运行 3D IDA 分析。

    强度指标（IM）取两分量在 T₁ 处谱加速度的几何均值：

        Sa_gm = sqrt(Sa_X × Sa_Y)

    两分量使用相同的缩放系数：SF = IM_target / Sa_gm。

    对于平面结构，将同一结构模型分别在 X 和 Y 激励下独立运行，
    假设两方向具有相同的结构体系。

    Parameters
    ----------
    FEModel : IDAModelProtocol
        用于非线性动力分析的有限元模型。
    IM_list : list
        目标强度指标值（几何均值 Sa，单位 g）。
    EQRecordX : str
        X 方向地震动记录文件路径。
    EQRecordY : str
        Y 方向地震动记录文件路径。
    period : float
        用于计算谱加速度的基本周期（s）。
    DeltaT : str or float, optional
        动力分析时间步长，使用 'AsInRecord' 则采用记录自身步长。
    _status_queue : multiprocessing.Queue, optional
        多进程状态队列，用于向主进程上报 IM 进度。

    Returns
    -------
    pandas.DataFrame
        一个记录对所有 IM 级别的 3D IDA 分析结果。
    """
    pair_name     = Path(EQRecordX).stem + '+' + Path(EQRecordY).stem
    record_name_X = Path(EQRecordX).stem
    record_name_Y = Path(EQRecordY).stem

    _, _, Sa_gm = _sa_geomean(EQRecordX, EQRecordY, period)

    # 计算两分量原始（未缩放）PGV (m/s)
    PGV_X_original = _compute_pgv(EQRecordX)
    PGV_Y_original = _compute_pgv(EQRecordY)

    IDA_result = pd.DataFrame()
    n_im = len(IM_list)

    if _status_queue is None:
        im_iter = tqdm(
            enumerate(IM_list), total=n_im,
            desc=f"  {pair_name[:28]}", leave=False, unit='IM',
        )
    else:
        im_iter = enumerate(IM_list)

    for im_idx, IM in im_iter:
        # 多进程模式：第一条 IM 前先上报以创建子进度条
        if _status_queue is not None and im_idx == 0:
            _status_queue.put({
                'record': pair_name, 'im_idx': 0, 'im_total': n_im,
                'IM': IM, 'finished': True, 'tCurrent': 0.0, 'TotalTime': 0.0,
            })

        SF = IM / Sa_gm  # 两分量统一缩放系数

        # ── X 分量分析 ───────────────────────────────────────────────────
        FEModel_X = copy.deepcopy(FEModel)
        FEModel_X.UniqueRecorderPrefix = f'URP_X_{record_name_X}_'
        Iffinish_X, tCurrent_X, TotalTime = FEModel_X.DynamicAnalysis(
            str(EQRecordX), SF, False, DeltaT)

        # ── Y 分量分析 ───────────────────────────────────────────────────
        FEModel_Y = copy.deepcopy(FEModel)
        FEModel_Y.UniqueRecorderPrefix = f'URP_Y_{record_name_Y}_'
        Iffinish_Y, tCurrent_Y, _ = FEModel_Y.DynamicAnalysis( str(EQRecordY), SF, False, DeltaT)

        Iffinish = bool(Iffinish_X) and bool(Iffinish_Y)

        if _status_queue is not None:
            _status_queue.put({
                'record': pair_name, 'im_idx': im_idx + 1, 'im_total': n_im,
                'IM': IM, 'finished': Iffinish,
                'tCurrent': max(tCurrent_X, tCurrent_Y),
                'TotalTime': TotalTime,
            })
        else:
            postfix = (
                f"{'FAIL' if not Iffinish else 'OK'} "
                f"X={'OK' if Iffinish_X else 'FAIL'} "
                f"Y={'OK' if Iffinish_Y else 'FAIL'}"
            )
            im_iter.set_postfix_str(postfix)
            if not Iffinish:
                tqdm.write(
                    f"  [{pair_name}] IM={IM:.3f}g: "
                    f"X={'OK' if Iffinish_X else 'FAIL'} "
                    f"Y={'OK' if Iffinish_Y else 'FAIL'}"
                )

        data = {
            'IM':            IM,
            'EQRecord_X':    EQRecordX,
            'EQRecord_Y':    EQRecordY,
            'MaxDrift_X':    [list(FEModel_X.MaxDrift)],
            'MaxDrift_Y':    [list(FEModel_Y.MaxDrift)],
            'MaxAbsAccel_X': [list(FEModel_X.MaxAbsAccel)],
            'MaxAbsAccel_Y': [list(FEModel_Y.MaxAbsAccel)],
            'MaxAbsVel_X':   [list(getattr(FEModel_X, 'MaxAbsVel', []))],
            'MaxAbsVel_Y':   [list(getattr(FEModel_Y, 'MaxAbsVel', []))],
            'PGV_X':         PGV_X_original * (IM / Sa_gm) if Sa_gm > 0 else 0.0,
            'PGV_Y':         PGV_Y_original * (IM / Sa_gm) if Sa_gm > 0 else 0.0,
            'ResDrift_X':    _to_float(FEModel_X.ResDrift),
            'ResDrift_Y':    _to_float(FEModel_Y.ResDrift),
            'Iffinish':      Iffinish,
            'Iffinish_X':    bool(Iffinish_X),
            'Iffinish_Y':    bool(Iffinish_Y),
            'tCurrent_X':    tCurrent_X,
            'tCurrent_Y':    tCurrent_Y,
            'TotalTime':     TotalTime,
        }
        # 追加用户自定义 EDP 列（X 分量和 Y 分量）
        if ExtraEDP:
            for _edp_name, _attr_name in ExtraEDP.items():
                _val_x = getattr(FEModel_X, _attr_name, None)
                _val_y = getattr(FEModel_Y, _attr_name, None)
                for _suffix, _val in [('_X', _val_x), ('_Y', _val_y)]:
                    _col = _edp_name + _suffix
                    if _val is None:
                        data[_col] = float('nan')
                    elif hasattr(_val, '__len__') and not isinstance(_val, str):
                        data[_col] = [list(_val)]
                    else:
                        data[_col] = float(_val)
        IDA_result = pd.concat([IDA_result, pd.DataFrame(data)], ignore_index=True)

    return IDA_result


# ────────────────────────────────────────────── 多记录对批量分析（支持并行）


def _IDA3D_f(
    FEModel: IDAModelProtocol,
    IM_list: list,
    period: float,
    EQRecordfile_pair_list: list = None,
    DeltaT='AsInRecord',
    NumPool: int = 1,
    ExtraEDP=None,
) -> pd.DataFrame:
    """对多个地震动记录对并行执行 3D IDA 分析。

    Parameters
    ----------
    FEModel : IDAModelProtocol
        基础有限元模型。
    IM_list : list
        目标强度指标值（几何均值 Sa，单位 g）。
    period : float
        用于计算谱加速度的基本周期（s）。
    EQRecordfile_pair_list : list of (str, str), optional
        形如 [(X_path, Y_path), ...] 的记录对列表。
        默认使用 FEMA P-695 远场地震动记录对（MetaData.txt）。
    DeltaT : str or float, optional
        动力分析时间步长。
    NumPool : int, optional
        工作进程数，设为 1 时串行执行。

    Returns
    -------
    pandas.DataFrame
        所有记录对的 3D IDA 结果。
    """
    if EQRecordfile_pair_list is None:
        from .. import __file__ as _mdof_file
        fema_dir  = Path(_mdof_file).parent / 'Resources' / 'FEMA_P-695_far-field_ground_motions'
        meta_path = fema_dir / 'MetaData.txt'
        if not meta_path.exists():
            raise FileNotFoundError(f"未找到 FEMA P-695 元数据文件：{meta_path}")
        T = pd.read_table(str(meta_path), sep=',')
        EQRecordfile_pair_list = [
            (
                str(fema_dir / row['AccelXfile'].replace('.txt', '')),
                str(fema_dir / row['AccelYfile'].replace('.txt', '')),
            )
            for _, row in T.iterrows()
        ]

    # 初始化空结果 DataFrame（保证列一致性）
    IDA_result = pd.DataFrame({
        col: []
        for col in [
            'IM', 'EQRecord_X', 'EQRecord_Y',
            'MaxDrift_X', 'MaxDrift_Y',
            'MaxAbsAccel_X', 'MaxAbsAccel_Y',
            'MaxAbsVel_X', 'MaxAbsVel_Y',
            'ResDrift_X', 'ResDrift_Y',
            'Iffinish', 'Iffinish_X', 'Iffinish_Y',
            'tCurrent_X', 'tCurrent_Y', 'TotalTime',
        ]
    })

    total = len(EQRecordfile_pair_list)

    if NumPool == 1:
        for eq_x, eq_y in tqdm(EQRecordfile_pair_list, desc='IDA 3D', unit='pair', total=total):
            FEModel_ = copy.deepcopy(FEModel)
            result   = _IDA3D_1record_pair(FEModel_, IM_list, eq_x, eq_y, period, DeltaT, ExtraEDP=ExtraEDP)
            IDA_result = pd.concat([IDA_result, result], ignore_index=True)
    else:
        with mp.Manager() as manager:
            status_queue = manager.Queue()
            stop_event   = threading.Event()

            def _progress_display(queue, stop_event):
                available_positions = list(range(1, NumPool + 1))
                record_bars = {}

                while not stop_event.is_set() or not queue.empty():
                    try:
                        msg        = queue.get(timeout=0.2)
                        record     = msg['record']
                        im_idx     = msg['im_idx']
                        im_total   = msg['im_total']

                        if record not in record_bars:
                            pos = available_positions.pop(0) if available_positions else NumPool
                            bar = tqdm(
                                total=im_total,
                                desc=f"  {record[:25]}",
                                position=pos, leave=False, unit='IM',
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]',
                            )
                            record_bars[record] = (bar, pos)

                        bar, pos = record_bars[record]
                        bar.n = im_idx
                        bar.refresh()

                        if im_idx >= im_total:
                            bar.close()
                            del record_bars[record]
                            available_positions.append(pos)
                            available_positions.sort()
                    except Exception:
                        pass

                for bar, _ in list(record_bars.values()):
                    bar.close()

            disp_thread = threading.Thread(
                target=_progress_display, args=(status_queue, stop_event), daemon=True,
            )
            disp_thread.start()

            with mp.Pool(NumPool) as pool:
                futures = [
                    pool.apply_async(
                        _IDA3D_1record_pair,
                        args=(copy.deepcopy(FEModel), IM_list, eq_x, eq_y, period, DeltaT, status_queue),
                        kwds={'ExtraEDP': ExtraEDP},
                    )
                    for eq_x, eq_y in EQRecordfile_pair_list
                ]
                with tqdm(total=total, desc='IDA 3D', unit='pair', position=0) as pbar:
                    for future in futures:
                        IDA_result = pd.concat([IDA_result, future.get()], ignore_index=True)
                        pbar.update(1)

            stop_event.set()
            disp_thread.join(timeout=3.0)

    return IDA_result


# ──────────────────────────────────────────────────────── CSV 读写工具


def _read_IDA3D_results_csv(csv_file: Union[str, Path]) -> pd.DataFrame:
    """读取 3D IDA CSV 并将字符串形式的数组列还原为 numpy 数组。"""
    def _parse(value):
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text == '' or text.lower() == 'nan':
            return np.array([])
        return np.fromstring(text.strip('[]').replace(',', ' '), sep=' ')

    array_cols = [
        'MaxDrift_X', 'MaxDrift_Y',
        'MaxAbsAccel_X', 'MaxAbsAccel_Y',
        'MaxAbsVel_X', 'MaxAbsVel_Y',
    ]
    converters = {col: _parse for col in array_cols}
    df = pd.read_csv(Path(csv_file), converters=converters)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # 自动解析额外 EDP 列（非标准列中含 '[' 的字符串值解析为数组）
    for col in df.columns:
        if col in _IDA3D_STANDARD_COLS:
            continue
        if df[col].dtype == object:
            try:
                df[col] = df[col].apply(
                    lambda v: _parse(v) if isinstance(v, str) and '[' in v else v
                )
            except Exception:
                pass
    return df


# ─────────────────────────────────────────── EDP 插值（供损失评估使用）


def interp_edp_from_ida_3D(
    ida_3d_csv: Union[str, Path, 'pd.DataFrame'],
    im_target: float,
    num_stories: int,
):
    """从 3D IDA 结果 CSV 中提取目标 IM 水平处各记录对的原始 EDP 样本。

    在相邻 IM 级别之间对每条记录对进行线性插值。

    Parameters
    ----------
    ida_3d_csv : str or Path
        3D IDA 结果 CSV 路径（由 IDA3DAnalysis.SaveToCSV 输出）。
    im_target : float
        目标地震动强度（几何均值 Sa，单位 g）。
    num_stories : int
        楼层数。

    Returns
    -------
    drift_X : np.ndarray, shape (n_pairs, num_stories)
        X 方向各层最大层间位移角（rad）。
    drift_Y : np.ndarray, shape (n_pairs, num_stories)
        Y 方向各层最大层间位移角（rad）。
    accel_X : np.ndarray, shape (n_pairs, num_stories+1)
        X 方向各楼面最大绝对加速度（g），第 0 列为地面层 PGA。
    accel_Y : np.ndarray, shape (n_pairs, num_stories+1)
        Y 方向各楼面最大绝对加速度（g），第 0 列为地面层 PGA。
    res_X : np.ndarray, shape (n_pairs,)
        X 方向全楼最大残余层间位移角（rad）。
    res_Y : np.ndarray, shape (n_pairs,)
        Y 方向全楼最大残余层间位移角（rad）。
    vel_X : np.ndarray, shape (n_pairs, num_stories+1)
        X 方向各楼面最大绝对速度（m/s），第 0 列为地面层。
    vel_Y : np.ndarray, shape (n_pairs, num_stories+1)
        Y 方向各楼面最大绝对速度（m/s），第 0 列为地面层。
    """
    N  = int(num_stories)
    if isinstance(ida_3d_csv, pd.DataFrame):
        df = ida_3d_csv.copy()
        # 确保数组列已解析为 numpy 数组
        _array_cols = ['MaxDrift_X', 'MaxDrift_Y', 'MaxAbsAccel_X', 'MaxAbsAccel_Y', 'MaxAbsVel_X', 'MaxAbsVel_Y']
        for _col in _array_cols:
            if _col in df.columns:
                df[_col] = df[_col].apply(
                    lambda v: _parse_ida_array(v) if isinstance(v, str) else np.asarray(v, dtype=float)
                )
    else:
        df = _read_IDA3D_results_csv(ida_3d_csv)
    df['Iffinish'] = df['Iffinish'].astype(bool)
    df = df.loc[df['Iffinish'], :].reset_index(drop=True)

    if df.empty:
        raise ValueError("3D IDA 结果中没有收敛完成的记录，无法提取 EDP。")

    drift_X_rows, drift_Y_rows = [], []
    accel_X_rows, accel_Y_rows = [], []
    res_X_rows,   res_Y_rows   = [], []
    vel_X_rows,   vel_Y_rows   = [], []
    pgv_X_rows,   pgv_Y_rows   = [], []

    # 检测额外自定义 EDP 列（以 _X 后缀且不在标准列集合中）
    _extra_base_names = [
        col[:-2] for col in df.columns
        if col not in _IDA3D_STANDARD_COLS and col.endswith('_X')
    ]
    _extra_x_rows = {name: [] for name in _extra_base_names}
    _extra_y_rows = {name: [] for name in _extra_base_names}

    has_vel = 'MaxAbsVel_X' in df.columns and 'MaxAbsVel_Y' in df.columns
    has_pgv = 'PGV_X' in df.columns and 'PGV_Y' in df.columns

    df['_pair'] = df['EQRecord_X'].astype(str) + '|' + df['EQRecord_Y'].astype(str)
    for _, rows in df.groupby('_pair', sort=False):
        if rows.empty:
            continue

        # 漂移
        drift_x = np.asarray(_interp_ida_value(rows, im_target, 'MaxDrift_X'), dtype=float)[:N]
        drift_y = np.asarray(_interp_ida_value(rows, im_target, 'MaxDrift_Y'), dtype=float)[:N]

        # 加速度（mm/s² → g），拼接地面层
        ax_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsAccel_X'), dtype=float)[:N]
        ay_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsAccel_Y'), dtype=float)[:N]
        ground = np.array([im_target * 0.4])
        accel_x = np.concatenate([ground, ax_raw / 9800.0])
        accel_y = np.concatenate([ground, ay_raw / 9800.0])

        # 残余位移
        res_x = float(np.asarray(_interp_ida_value(rows, im_target, 'ResDrift_X'), dtype=float))
        res_y = float(np.asarray(_interp_ida_value(rows, im_target, 'ResDrift_Y'), dtype=float))

        # 速度（mm/s → m/s），拼接地面层
        if has_vel:
            vx_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsVel_X'), dtype=float)
            vy_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsVel_Y'), dtype=float)
            if len(vx_raw) >= N + 1:
                vel_x = vx_raw[:N + 1] / 1000.0
            else:
                vel_x = np.concatenate([[0.0], vx_raw[:N]]) / 1000.0
            if len(vy_raw) >= N + 1:
                vel_y = vy_raw[:N + 1] / 1000.0
            else:
                vel_y = np.concatenate([[0.0], vy_raw[:N]]) / 1000.0
        else:
            vel_x = np.full(N + 1, 10.0)
            vel_y = np.full(N + 1, 10.0)

        drift_X_rows.append(drift_x)
        drift_Y_rows.append(drift_y)
        accel_X_rows.append(accel_x)
        accel_Y_rows.append(accel_y)
        res_X_rows.append(res_x)
        res_Y_rows.append(res_y)
        vel_X_rows.append(vel_x)
        vel_Y_rows.append(vel_y)

        if has_pgv:
            pgv_X_rows.append(float(_interp_ida_value(rows, im_target, 'PGV_X')))
            pgv_Y_rows.append(float(_interp_ida_value(rows, im_target, 'PGV_Y')))

        # 提取自定义 EDP
        for _name in _extra_base_names:
            _col_x = _name + '_X'
            _col_y = _name + '_Y'
            if _col_x in rows.columns:
                _extra_x_rows[_name].append(
                    np.asarray(_interp_ida_value(rows, im_target, _col_x), dtype=float))
            if _col_y in rows.columns:
                _extra_y_rows[_name].append(
                    np.asarray(_interp_ida_value(rows, im_target, _col_y), dtype=float))

    drift_X = np.clip(np.array(drift_X_rows, dtype=float), 1e-8, None)
    drift_Y = np.clip(np.array(drift_Y_rows, dtype=float), 1e-8, None)
    accel_X = np.clip(np.array(accel_X_rows, dtype=float), 1e-8, None)
    accel_Y = np.clip(np.array(accel_Y_rows, dtype=float), 1e-8, None)
    res_X   = np.clip(np.array(res_X_rows,   dtype=float), 1e-8, None)
    res_Y   = np.clip(np.array(res_Y_rows,   dtype=float), 1e-8, None)
    vel_X   = np.clip(np.array(vel_X_rows,   dtype=float), 1e-8, None)
    vel_Y   = np.clip(np.array(vel_Y_rows,   dtype=float), 1e-8, None)

    pgv_X = np.clip(np.array(pgv_X_rows, dtype=float), 1e-8, None) if pgv_X_rows else None
    pgv_Y = np.clip(np.array(pgv_Y_rows, dtype=float), 1e-8, None) if pgv_Y_rows else None

    extra_x_dict: dict = {}
    extra_y_dict: dict = {}
    for _name, _vals in _extra_x_rows.items():
        if _vals:
            try:
                extra_x_dict[_name] = np.array(_vals, dtype=float)
            except (ValueError, TypeError):
                pass
    for _name, _vals in _extra_y_rows.items():
        if _vals:
            try:
                extra_y_dict[_name] = np.array(_vals, dtype=float)
            except (ValueError, TypeError):
                pass

    return drift_X, drift_Y, accel_X, accel_Y, res_X, res_Y, vel_X, vel_Y, pgv_X, pgv_Y, extra_x_dict, extra_y_dict


# ──────────────────────────────────────────────── 2D 包络转换（Hazus 兼容）


def IDA3D_to_2d_envelope(ida_3d_csv: Union[str, Path]) -> pd.DataFrame:
    """将 3D IDA 结果转换为 2D 等效包络（取 X/Y 逐层最大值）。

    结果格式与标准 IDA.py 输出兼容，可直接传入 Hazus 损失评估函数或
    其他仅支持单向 EDP 的下游模块。

    Parameters
    ----------
    ida_3d_csv : str or Path
        3D IDA 结果 CSV 文件路径。

    Returns
    -------
    pd.DataFrame
        2D 包络 IDA 结果，列含 IM, EQRecord, MaxDrift, MaxAbsAccel,
        MaxAbsVel, ResDrift, Iffinish, tCurrent, TotalTime。
    """
    df = _read_IDA3D_results_csv(ida_3d_csv)

    def _env(a, b):
        a_ = _parse_ida_array(a) if isinstance(a, str) else np.asarray(a, dtype=float)
        b_ = _parse_ida_array(b) if isinstance(b, str) else np.asarray(b, dtype=float)
        if a_.size == 0:
            return b_
        if b_.size == 0:
            return a_
        n = min(len(a_), len(b_))
        return np.maximum(a_[:n], b_[:n])

    rows = []
    for _, row in df.iterrows():
        drift = _env(row['MaxDrift_X'],    row['MaxDrift_Y'])
        accel = _env(row['MaxAbsAccel_X'], row['MaxAbsAccel_Y'])

        vel_x = row.get('MaxAbsVel_X', np.array([]))
        vel_y = row.get('MaxAbsVel_Y', np.array([]))
        vel   = _env(vel_x, vel_y) if 'MaxAbsVel_X' in df.columns else np.array([])

        res_drift = max(float(row['ResDrift_X']), float(row['ResDrift_Y']))

        rows.append({
            'IM':          row['IM'],
            'EQRecord':    row['EQRecord_X'],
            'MaxDrift':    list(drift),
            'MaxAbsAccel': list(accel),
            'MaxAbsVel':   list(vel) if vel.size > 0 else [],
            'ResDrift':    res_drift,
            'Iffinish':    bool(row['Iffinish']),
            'tCurrent':    float(row.get('tCurrent_X', 0.0)),
            'TotalTime':   float(row.get('TotalTime', 0.0)),
        })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────── 绘图工具


def plot_IDA3D_results(
    IDA_result: Union[str, Path, pd.DataFrame],
    Stat: bool = True,
    FigName: Union[str, Path] = 'IDA_3D.jpg',
):
    """绘制 3D IDA 曲线图（横轴为 X/Y 两方向最大层间位移角的包络值）。

    Parameters
    ----------
    IDA_result : str, Path, or pd.DataFrame
        3D IDA 结果：CSV 路径或已读取的 DataFrame。
    Stat : bool
        False 时绘制各记录对的单条曲线；True 时绘制统计包络（中位数 ± σ）。
    FigName : str or Path
        输出图片路径（jpg）。
    """
    if isinstance(IDA_result, (str, Path)):
        IDA_result = _read_IDA3D_results_csv(IDA_result)

    def _max_drift_env(row):
        """取 X/Y 两方向最大层间位移角的包络（逐层 max，再取全楼 max）。"""
        def _arr(val):
            a = _parse_ida_array(val) if isinstance(val, str) else np.asarray(val, dtype=float)
            return float(a.max()) if a.size > 0 else 0.0
        return max(_arr(row['MaxDrift_X']), _arr(row['MaxDrift_Y']))

    fig, ax = plt.subplots(figsize=(7, 5))
    pair_list = list(Counter(IDA_result['EQRecord_X'].values).keys())

    if not Stat:
        for eq_x in pair_list:
            rows = IDA_result[IDA_result['EQRecord_X'] == eq_x].sort_values('IM')
            ax.plot([_max_drift_env(r) for _, r in rows.iterrows()], rows['IM'].values)
    else:
        for i, eq_x in enumerate(pair_list):
            rows = IDA_result[IDA_result['EQRecord_X'] == eq_x].sort_values('IM')
            ax.plot(
                [_max_drift_env(r) for _, r in rows.iterrows()],
                rows['IM'].values,
                color='0.75', linewidth=0.6, alpha=0.8,
                label='Records' if i == 0 else None, zorder=1,
            )

        IM_list_sorted = sorted(Counter(IDA_result['IM'].values.tolist()).keys())
        medians, lo, hi = [], [], []
        for im in IM_list_sorted:
            sub  = IDA_result[IDA_result['IM'] == im]
            vals = [_max_drift_env(r) for _, r in sub.iterrows()]
            ln   = np.log(np.clip(vals, 1e-12, None))
            med  = np.exp(np.mean(ln))
            sig  = np.std(ln)
            medians.append(med)
            lo.append(np.exp(np.mean(ln) - sig))
            hi.append(np.exp(np.mean(ln) + sig))

        ax.plot(medians, IM_list_sorted, 'k',  label='Median', zorder=3)
        ax.plot(lo,      IM_list_sorted, 'b',  label='-σ',     zorder=3)
        ax.plot(hi,      IM_list_sorted, 'g',  label='+σ',     zorder=3)

    ax.set_xlabel('Max drift ratio (envelope of X & Y)',
                  fontdict={'family': 'Times New Roman', 'size': 12})
    ax.set_ylabel('Sa_gm (g)',
                  fontdict={'family': 'Times New Roman', 'size': 12})
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 11})

    plt.tight_layout()
    plt.savefig(FigName, dpi=600, format='jpg', bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────── 高级接口封装类


class IDA3DAnalysis:
    """基于双向地震动分量的 3D IDA 高级接口封装类。

    用法示例::

        from MDOFModel.analysis import IDA_3D

        ida3d = IDA_3D.IDA3DAnalysis(wrapper_model)
        ida3d.Analyze(IM_list=[0.1, 0.3, 0.5, 1.0, 2.0], NumPool=8)
        ida3d.SaveToCSV('IDA_results_3D.csv')
        IDA_3D.plot_IDA3D_results('IDA_results_3D.csv', Stat=True)
    """

    FEModel:    IDAModelProtocol = None
    IDA_result: pd.DataFrame     = None

    def __init__(self, FEModel: IDAModelProtocol):
        """
        Parameters
        ----------
        FEModel : IDAModelProtocol
            在分析中使用的有限元模型对象。
        """
        self.FEModel = FEModel

    def Analyze(
        self,
        IM_list: list,
        period: float = None,
        EQRecordfile_pair_list: list = None,
        DeltaT='AsInRecord',
        NumPool: int = 1,
        ExtraEDP=None,
    ) -> pd.DataFrame:
        """执行 3D IDA 分析并保存结果。

        Parameters
        ----------
        IM_list : list
            目标 IM 值列表（几何均值 Sa，单位 g）。
        period : float, optional
            计算谱加速度的周期（s）。若为 None，从模型的 T1 属性读取。
        EQRecordfile_pair_list : list of (str, str), optional
            [(X_path, Y_path), ...] 形式的记录对列表。
            默认使用 FEMA P-695 远场地震动所有记录对。
        DeltaT : str or float, optional
            动力分析时间步长。
        NumPool : int, optional
            并行进程数，设为 1 时串行执行。
        ExtraEDP : dict, optional
            自定义 EDP 配置，格式为 ``{'EDP类型名': '模型属性名'}``。
            例如 ``{'STRAIN': 'MaxColStrain'}`` 会分别读取
            ``FEModel_X.MaxColStrain`` 和 ``FEModel_Y.MaxColStrain``，
            并以 ``STRAIN_X`` 和 ``STRAIN_Y`` 列写入结果。

        Returns
        -------
        pd.DataFrame
            所有记录对和 IM 级别的 3D IDA 结果。
        """
        if period is None:
            if hasattr(self.FEModel, 'T1'):
                period = float(self.FEModel.T1)
            else:
                raise ValueError("Model 需要 T1 属性，或通过 period 参数手动指定。")

        self.IDA_result = _IDA3D_f(
            self.FEModel, IM_list, period,
            EQRecordfile_pair_list, DeltaT, NumPool, ExtraEDP=ExtraEDP,
        )
        return self.IDA_result

    def SaveToCSV(self, OutputCSVFile: Union[str, Path]) -> None:
        """将 3D IDA 分析结果保存为 CSV 文件。

        Parameters
        ----------
        OutputCSVFile : str or Path
            输出 CSV 文件路径。
        """
        if self.IDA_result is None:
            raise RuntimeError("尚未执行 IDA 分析，请先调用 Analyze() 方法。")
        self.IDA_result.to_csv(Path(OutputCSVFile), index=False, encoding='utf-8-sig')

    @staticmethod
    def plot_IDA_results(
        IDA_result: Union[str, Path, pd.DataFrame],
        Stat: bool = True,
        FigName: Union[str, Path] = 'IDA_3D.jpg',
    ):
        """绘制 3D IDA 曲线图。委托给模块级 plot_IDA3D_results 函数。"""
        return plot_IDA3D_results(IDA_result, Stat=Stat, FigName=FigName)
