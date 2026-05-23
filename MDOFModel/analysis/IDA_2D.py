########################################################
# IDA_2D.py – 平面结构增量动力分析（单向 / 双向地震动）
#
# 核心设计：
#   - 单向分析：records = ['file1.AT2', 'file2.AT2', ...]
#   - 双向分析：records = [('file_x.AT2', 'file_y.AT2'), ...]
#   两种模式均通过 IDA_1record / IDA_f / IDAAnalysis 统一接受，
#   区别仅在于 record_y 参数（None 表示单向，非 None 表示双向）。
#
# 双向 IM 定义：Sa_gm = sqrt(Sa_X × Sa_Y)（两分量几何均值）
#
# 依赖：openseespy, pandas, numpy, eqsig, tqdm
########################################################

import copy
import math
import multiprocessing as mp
import threading
from collections import Counter
from pathlib import Path
from typing import Protocol, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import ReadRecord
from ..utils.record_utils import compute_sa as _compute_sa, compute_pgv as _compute_pgv

# ── 模型协议 & 标准列集合 ─────────────────────────────────────────────────────

class IDAModelProtocol(Protocol):
    UniqueRecorderPrefix: str
    MaxDrift: list
    MaxAbsAccel: list
    MaxRelativeAccel: list
    MaxAbsVel: list
    ResDrift: list
    T1: float

    def DynamicAnalysis(self, record_file: str, scale_factor: float, ifprint: bool, delta_t) -> Tuple[bool, float, float]: ...

# 用于识别自定义 EDP 列（排除这些标准列后即为自定义列）
_STANDARD_COLS_1D = frozenset({
    'IM', 'EQRecord', 'MaxDrift', 'MaxAbsAccel', 'MaxRelativeAccel', 'MaxAbsVel', 'ResDrift', 'PGV', 'Iffinish', 'tCurrent', 'TotalTime',
})
_STANDARD_COLS_2D = frozenset({
    'IM', 'EQRecord_X', 'EQRecord_Y',
    'MaxDrift_X', 'MaxDrift_Y', 'MaxAbsAccel_X', 'MaxAbsAccel_Y', 'MaxAbsVel_X', 'MaxAbsVel_Y', 'PGV_X', 'PGV_Y', 'ResDrift_X', 'ResDrift_Y', 'Iffinish', 'Iffinish_X', 'Iffinish_Y', 'tCurrent_X', 'tCurrent_Y', 'TotalTime', '_pair',
})


# ── 地震动文件工具 ─────────────────────────────────────────────────────────────

def load_fema_records(bidir: bool = False) -> list:
    """返回 FEMA P-695 远场地震动记录列表。

    Parameters
    ----------
    bidir : bool
        False（默认）返回 X 分量路径列表；
        True 返回 ``(X_path, Y_path)`` 元组列表。
    """
    from .. import __file__ as _pkg
    fema_dir = Path(_pkg).parent / 'Resources' / 'FEMA_P-695_far-field_ground_motions'
    meta = fema_dir / 'MetaData.txt'
    if not meta.exists():
        raise FileNotFoundError(f"FEMA P-695 元数据文件未找到：{meta}")
    T = pd.read_table(str(meta), sep=',')
    if bidir:
        return [
            (str(fema_dir / row['AccelXfile'].replace('.txt', '')),
             str(fema_dir / row['AccelYfile'].replace('.txt', '')))
            for _, row in T.iterrows()
        ]
    return [str(fema_dir / x.replace('.txt', '')) for x in T['AccelXfile']]


# ── 数值插值工具 ───────────────────────────────────────────────────────────────

def _parse_ida_array(value) -> np.ndarray:
    """将字符串或已有数组统一解析为 numpy 数组。

    IDA CSV 中各层 EDP 以字符串形式存储，例如 ``"[0.01 0.02 0.03]"``
    或 ``"[0.01, 0.02, 0.03]"``。此函数统一处理字符串与已有数组两种输入，
    供插值函数在读取 CSV 后使用。

    Parameters
    ----------
    value : str or array-like
        待解析的值。字符串支持方括号包围、逗号或空格分隔，
        例如 ``"[0.01, 0.02]"``、``"0.01 0.02"``；
        非字符串输入将直接转换为 ``float64`` 数组。

    Returns
    -------
    numpy.ndarray
        解析后的一维浮点数组。
    """
    if isinstance(value, str):
        return np.fromstring(value.strip().strip('[]').replace(',', ' '), sep=' ')
    return np.asarray(value, dtype=float)


def _interp_ida_value(rows: pd.DataFrame, im_target: float, column: str):
    """在目标 IM 处对单条记录的 IDA 结果进行分段线性插值。

    对同一地震动记录在各 IM 级别下的分析结果，在目标 IM 处插值
    指定 EDP 列。超出 IM 范围时返回边界值（不外推）。
    目标列可为标量列（如 ``ResDrift``）或数组列（如 ``MaxDrift``）。

    Parameters
    ----------
    rows : pandas.DataFrame
        同一地震动记录的 IDA 结果子集，须包含 ``'IM'`` 列及 ``column`` 列。
    im_target : float
        目标强度指标值（单位：g）。
    column : str
        待插值的列名。

    Returns
    -------
    float or numpy.ndarray
        插值结果。标量列返回 float；数组列（各层 EDP）返回
        插值后的 numpy 数组，长度与原数组相同。
    """
    im_vals = rows['IM'].astype(float).to_numpy()
    order = np.argsort(im_vals)
    rows, im_vals = rows.iloc[order].reset_index(drop=True), im_vals[order]

    exact = np.where(np.isclose(im_vals, im_target))[0]
    if len(exact):
        return rows.iloc[exact[0]][column]
    if im_target <= im_vals[0]:
        return rows.iloc[0][column]
    if im_target >= im_vals[-1]:
        return rows.iloc[-1][column]

    up = int(np.searchsorted(im_vals, im_target, side='right'))
    w = (im_target - im_vals[up - 1]) / (im_vals[up] - im_vals[up - 1])
    lo_arr  = _parse_ida_array(rows.iloc[up - 1][column])
    hi_arr  = _parse_ida_array(rows.iloc[up][column])
    return lo_arr + w * (hi_arr - lo_arr)


def _to_scalar(val) -> float:
    """将标量或数组形式的 EDP 转换为单个浮点数（取各层最大值）。

    ``ResDrift`` 等 EDP 在部分模型中以各层列表返回，
    此函数将其归一化为单个代表值，取所有元素中的最大值。

    Parameters
    ----------
    val : float, list, or numpy.ndarray
        待转换的值。数组/列表取所有元素最大值；标量直接转换为 float。

    Returns
    -------
    float
        转换后的标量；空数组返回 ``0.0``。
    """
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val, dtype=float)
        return float(arr.max()) if arr.size > 0 else 0.0
    return float(val)


# ── 多进程进度显示线程 ─────────────────────────────────────────────────────────

def _progress_thread(queue, stop_event, pbar, num_workers: int):
    """多进程模式下的进度条显示线程，由主进程通过 ``threading.Thread`` 启动。

    持续监听状态队列，为每条地震动记录动态创建子进度条（显示当前
    IM 进度），同时更新外层主进度条（显示总 Sa 计算进度）。
    每条记录全部 IM 完成后自动关闭其子进度条并释放显示位置。

    Parameters
    ----------
    queue : multiprocessing.Queue
        状态消息队列，由各子进程的 ``IDA_1record`` 写入。
        每条消息为字典，包含以下字段：

        - ``record`` (str)：记录名（用于标识子进度条）
        - ``im_idx`` (int)：当前已完成的 IM 序号（0 表示初始化）
        - ``im_total`` (int)：该记录的 IM 总数
        - ``IM``, ``finished``, ``tCurrent``, ``TotalTime``：分析状态信息

    stop_event : threading.Event
        停止信号；主进程在所有子进程完成后调用 ``set()``，
        线程在队列清空后退出。
    pbar : tqdm.tqdm
        外层主进度条实例，对应所有记录 × IM 级别的总计数。
    num_workers : int
        并行进程数，决定各记录子进度条的 ``position`` 显示位置。
    """
    available, bars = list(range(1, num_workers + 1)), {}
    while not stop_event.is_set() or not queue.empty():
        try:
            msg = queue.get(timeout=0.2)
            rec, idx, total = msg['record'], msg['im_idx'], msg['im_total']
            if rec not in bars:
                pos = available.pop(0) if available else num_workers
                bars[rec] = (
                    tqdm(total=total, desc=f"  {rec[:25]}", position=pos,
                         leave=False, unit='IM',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'),
                    pos,
                )
            bar, pos = bars[rec]
            bar.n = idx
            bar.refresh()
            if idx > 0:
                pbar.update(1)
            if idx >= total:
                bar.close()
                del bars[rec]
                available.append(pos)
                available.sort()
        except Exception:
            pass
    for bar, _ in list(bars.values()):
        bar.close()


# ── 核心分析：单条（或一对）记录 ──────────────────────────────────────────────

def IDA_1record(
    FEModel: IDAModelProtocol,
    IM_list: list,
    record_x: str,
    period: float,
    record_y: str = None,
    DeltaT='AsInRecord',
    _status_queue=None,
    ExtraEDP: dict = None,
    _main_pbar=None,
) -> pd.DataFrame:
    """对单条（或一对）地震动记录运行 IDA 分析。

    Parameters
    ----------
    record_x : str
        地震动记录路径（单向）或 X 方向记录路径（双向）。
    record_y : str or None
        None → 单向分析；提供路径 → 双向分析（IM 取几何均值 Sa）。
    ExtraEDP : dict, optional
        ``{'列名': '模型属性名'}``；双向时自动附加 ``_X`` / ``_Y`` 后缀。
    """
    bidir  = record_y is not None
    name_x = Path(record_x).stem
    rec_name = f"{name_x}+{Path(record_y).stem}" if bidir else name_x

    # 计算参考 Sa 及 PGV（双向取几何均值 Sa）
    sa_x   = _compute_sa(record_x, period)
    Sa_ref = (math.sqrt(sa_x * _compute_sa(record_y, period))
              if bidir else sa_x)
    if Sa_ref <= 0:
        Sa_ref = 1e-9
    pgv_x = _compute_pgv(record_x)
    pgv_y = _compute_pgv(record_y) if bidir else None

    if not bidir:
        FEModel.UniqueRecorderPrefix = 'URP' + Path(record_x).name + '_'

    n_im = len(IM_list)
    im_iter = (
        tqdm(enumerate(IM_list), total=n_im,
             desc=f"  {rec_name[:28]}", leave=False, unit='IM')
        if _status_queue is None else enumerate(IM_list)
    )

    IDA_result = pd.DataFrame()

    for im_idx, IM in im_iter:
        # 多进程模式：第一条 IM 前先上报，供主进程创建子进度条
        if _status_queue is not None and im_idx == 0:
            _status_queue.put({'record': rec_name, 'im_idx': 0, 'im_total': n_im,
                               'IM': IM, 'finished': True, 'tCurrent': 0.0, 'TotalTime': 0.0})

        SF = IM / Sa_ref

        if bidir:
            FEM_X = copy.deepcopy(FEModel)
            FEM_X.UniqueRecorderPrefix = f'URP_X_{name_x}_'
            ok_x, t_x, TotalTime = FEM_X.DynamicAnalysis(record_x, SF, False, DeltaT)

            FEM_Y = copy.deepcopy(FEModel)
            FEM_Y.UniqueRecorderPrefix = f'URP_Y_{Path(record_y).stem}_'
            ok_y, t_y, _           = FEM_Y.DynamicAnalysis(record_y, SF, False, DeltaT)

            finished = bool(ok_x) and bool(ok_y)
            t_cur    = max(t_x, t_y)
        else:
            ok, t_cur, TotalTime = FEModel.DynamicAnalysis(record_x, SF, False, DeltaT)
            finished = bool(ok)

        if _status_queue is not None:
            _status_queue.put({'record': rec_name, 'im_idx': im_idx + 1, 'im_total': n_im,
                               'IM': IM, 'finished': finished,
                               'tCurrent': t_cur, 'TotalTime': TotalTime})
        else:
            im_iter.set_postfix_str(f"{'OK' if finished else 'FAIL'} t={t_cur:.1f}/{TotalTime:.1f}s")
            if not finished:
                tqdm.write(f"  [{rec_name}] IM={IM:.3f}g: FAIL t={t_cur:.2f}/{TotalTime:.2f}s")

        if _main_pbar is not None:
            _main_pbar.update(1)

        # 组装结果行
        if bidir:
            data = {
                'IM': IM, 'EQRecord_X': record_x, 'EQRecord_Y': record_y,
                'MaxDrift_X':    [list(FEM_X.MaxDrift)],
                'MaxDrift_Y':    [list(FEM_Y.MaxDrift)],
                'MaxAbsAccel_X': [list(FEM_X.MaxAbsAccel)],
                'MaxAbsAccel_Y': [list(FEM_Y.MaxAbsAccel)],
                'MaxAbsVel_X':   [list(getattr(FEM_X, 'MaxAbsVel', []))],
                'MaxAbsVel_Y':   [list(getattr(FEM_Y, 'MaxAbsVel', []))],
                'PGV_X': pgv_x * SF, 'PGV_Y': pgv_y * SF,
                'ResDrift_X': _to_scalar(FEM_X.ResDrift),
                'ResDrift_Y': _to_scalar(FEM_Y.ResDrift),
                'Iffinish': finished, 'Iffinish_X': bool(ok_x), 'Iffinish_Y': bool(ok_y),
                'tCurrent_X': t_x, 'tCurrent_Y': t_y, 'TotalTime': TotalTime,
            }
            if ExtraEDP:
                for col, attr in ExtraEDP.items():
                    for suf, m in (('_X', FEM_X), ('_Y', FEM_Y)):
                        v = getattr(m, attr, None)
                        data[col + suf] = (
                            float('nan') if v is None
                            else [list(v)] if hasattr(v, '__len__') and not isinstance(v, str)
                            else float(v)
                        )
        else:
            data = {
                'IM': IM, 'EQRecord': record_x,
                'MaxDrift':         [FEModel.MaxDrift],
                'MaxAbsAccel':      [FEModel.MaxAbsAccel],
                'MaxRelativeAccel': [FEModel.MaxRelativeAccel],
                'MaxAbsVel':        [getattr(FEModel, 'MaxAbsVel', [])],
                'PGV': pgv_x * SF,
                'ResDrift': FEModel.ResDrift, 'Iffinish': finished,
                'tCurrent': t_cur, 'TotalTime': TotalTime,
            }
            if ExtraEDP:
                for col, attr in ExtraEDP.items():
                    v = getattr(FEModel, attr, None)
                    data[col] = (
                        float('nan') if v is None
                        else [list(v)] if hasattr(v, '__len__') and not isinstance(v, str)
                        else float(v)
                    )

        IDA_result = pd.concat([IDA_result, pd.DataFrame(data)], ignore_index=True)

    return IDA_result


# ── 批量分析（多条记录，支持并行） ────────────────────────────────────────────

def IDA_f(
    FEModel: IDAModelProtocol,
    IM_list: list,
    period: float,
    records: list = None,
    DeltaT='AsInRecord',
    NumPool: int = 1,
    ExtraEDP: dict = None,
    output_csv: Union[str, Path] = None,
    restart: bool = False,
) -> pd.DataFrame:
    """对多条记录（或记录对）批量执行 IDA，支持多进程并行与断点续算。

    Parameters
    ----------
    records : list or None
        - ``None``：使用 FEMA P-695 远场 X 方向记录（单向）
        - ``list[str]``：单向分析
        - ``list[(str, str)]``：双向分析（IM 取几何均值 Sa）
    output_csv : str or Path, optional
        结果输出路径。若提供：

        - 每完成一条记录后立即写入该文件；
        - 若文件已存在且 ``restart=False``，自动加载已有结果并跳过已完成记录（断点续算）；
        - 若 ``restart=True``，忽略已有文件，从头重新计算。
    restart : bool, default False
        ``True`` 时强制从头重算，忽略 ``output_csv`` 中的已有结果。
    """
    if records is None:
        records = load_fema_records(bidir=False)

    bidir = isinstance(records[0], (tuple, list))
    label = 'IDA (bidir)' if bidir else 'IDA'

    def _unpack(rec):
        return (rec[0], rec[1]) if bidir else (rec, None)

    # ── 断点续算：加载已完成记录 ─────────────────────────────────────────
    IDA_result = pd.DataFrame()
    done_keys: set = set()
    ckpt = Path(output_csv) if output_csv else None

    if ckpt is not None and not restart and ckpt.exists():
        try:
            IDA_result = read_IDA_csv(ckpt)
            n_im = len(IM_list)
            if bidir and 'EQRecord_X' in IDA_result.columns and 'EQRecord_Y' in IDA_result.columns:
                counts = IDA_result.groupby(['EQRecord_X', 'EQRecord_Y']).size()
                done_keys = {k for k, v in counts.items() if v >= n_im}
            elif not bidir and 'EQRecord' in IDA_result.columns:
                counts = IDA_result.groupby('EQRecord').size()
                done_keys = {k for k, v in counts.items() if v >= n_im}
            if done_keys:
                tqdm.write(f"  [断点续算] 已跳过 {len(done_keys)} 条记录（从 {ckpt.name} 加载）")
        except Exception as e:
            tqdm.write(f"  [断点续算] 读取断点文件失败（{e}），将从头开始")
            IDA_result = pd.DataFrame()
            done_keys = set()

    def _is_done(rec) -> bool:
        rx, ry = _unpack(rec)
        return (rx, ry) in done_keys if bidir else rx in done_keys

    def _save_ckpt() -> None:
        if ckpt is not None:
            IDA_result.to_csv(ckpt, index=False, encoding='utf-8-sig')

    pending = [rec for rec in records if not _is_done(rec)]
    if not pending:
        tqdm.write("  [断点续算] 所有记录已完成，直接返回已有结果。")
        return IDA_result

    total_all = len(records) * len(IM_list)
    done_sa   = len(done_keys) * len(IM_list)

    if NumPool == 1:
        with tqdm(total=total_all, initial=done_sa, desc=label, unit='Sa', position=0) as pbar:
            for rec in pending:
                rx, ry = _unpack(rec)
                result = IDA_1record(copy.deepcopy(FEModel), IM_list, rx, period, ry, DeltaT, None, ExtraEDP, pbar)
                IDA_result = pd.concat([IDA_result, result], ignore_index=True)
                _save_ckpt()
    else:
        with mp.Manager() as manager:
            sq       = manager.Queue()
            stop_ev  = threading.Event()
            with tqdm(total=total_all, initial=done_sa, desc=label, unit='Sa', position=0) as pbar:
                t = threading.Thread(target=_progress_thread,
                                     args=(sq, stop_ev, pbar, NumPool), daemon=True)
                t.start()
                with mp.Pool(NumPool) as pool:
                    futures = [
                        pool.apply_async(
                            IDA_1record,
                            args=(copy.deepcopy(FEModel), IM_list, _unpack(rec)[0], period, _unpack(rec)[1], DeltaT, sq),
                            kwds={'ExtraEDP': ExtraEDP},
                        )
                        for rec in pending
                    ]
                    for fut in futures:
                        IDA_result = pd.concat([IDA_result, fut.get()], ignore_index=True)
                        _save_ckpt()
            stop_ev.set()
            t.join(timeout=3.0)

    return IDA_result


# ── CSV 读写 ───────────────────────────────────────────────────────────────────

def read_IDA_csv(csv_file: Union[str, Path]) -> pd.DataFrame:
    """读取 IDA CSV（自动兼容单向/双向格式），将字符串数组列解析为 numpy 数组。"""
    def _parse(v):
        if not isinstance(v, str):
            return v
        t = v.strip()
        return np.array([]) if t == '' or t.lower() == 'nan' else \
            np.fromstring(t.strip('[]').replace(',', ' '), sep=' ')

    # 先读一行探测格式
    tmp = pd.read_csv(Path(csv_file), nrows=0)
    bidir = 'EQRecord_X' in tmp.columns

    array_cols = (
        ['MaxDrift_X', 'MaxDrift_Y', 'MaxAbsAccel_X', 'MaxAbsAccel_Y',
         'MaxAbsVel_X', 'MaxAbsVel_Y']
        if bidir else
        ['MaxDrift', 'MaxAbsAccel', 'MaxRelativeAccel', 'MaxAbsVel', 'ResDrift']
    )
    df = pd.read_csv(Path(csv_file), converters={c: _parse for c in array_cols})
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # 自动解析其余含 '[' 的字符串列（自定义 EDP）
    std_cols = _STANDARD_COLS_2D if bidir else _STANDARD_COLS_1D
    for col in df.columns:
        if col not in std_cols and df[col].dtype == object:
            try:
                df[col] = df[col].apply(
                    lambda v: _parse(v) if isinstance(v, str) and '[' in v else v)
            except Exception:
                pass
    return df


# ── 绘图（自动检测单向/双向） ──────────────────────────────────────────────────

def plot_IDA(
    IDA_result: Union[str, Path, pd.DataFrame],
    Stat: bool = True,
    FigName: Union[str, Path] = 'IDA.jpg',
):
    """绘制 IDA 曲线图（自动识别单向/双向格式）。

    Parameters
    ----------
    Stat : bool
        False 绘制各条记录曲线；True 绘制中位数 ± σ 统计包络。
    """
    if isinstance(IDA_result, (str, Path)):
        IDA_result = read_IDA_csv(IDA_result)

    bidir = 'EQRecord_X' in IDA_result.columns

    def _max_drift(row):
        if bidir:
            ax = _parse_ida_array(row['MaxDrift_X']) if isinstance(row['MaxDrift_X'], str) \
                 else np.asarray(row['MaxDrift_X'], dtype=float)
            ay = _parse_ida_array(row['MaxDrift_Y']) if isinstance(row['MaxDrift_Y'], str) \
                 else np.asarray(row['MaxDrift_Y'], dtype=float)
            return float(max(ax.max() if ax.size else 0, ay.max() if ay.size else 0))
        else:
            a = _parse_ida_array(row['MaxDrift']) if isinstance(row['MaxDrift'], str) \
                else np.asarray(row['MaxDrift'], dtype=float)
            return float(a.max()) if a.size else 0.0

    record_col = 'EQRecord_X' if bidir else 'EQRecord'
    ylabel     = 'Sa_gm (g)' if bidir else 'Spectral acceleration (g)'

    fig, ax = plt.subplots()
    groups = list(Counter(IDA_result[record_col].values).keys())

    if not Stat:
        for rec in groups:
            rows = IDA_result[IDA_result[record_col] == rec].sort_values('IM')
            ax.plot([_max_drift(r) for _, r in rows.iterrows()], rows['IM'].values)
    else:
        for i, rec in enumerate(groups):
            rows = IDA_result[IDA_result[record_col] == rec].sort_values('IM')
            ax.plot([_max_drift(r) for _, r in rows.iterrows()], rows['IM'].values,
                    color='0.75', linewidth=0.6, alpha=0.8,
                    label='Records' if i == 0 else None, zorder=1)

        IM_list_sorted = sorted(Counter(IDA_result['IM'].values.tolist()).keys())
        medians, lo, hi = [], [], []
        for im in IM_list_sorted:
            vals = [_max_drift(r) for _, r in IDA_result[IDA_result['IM'] == im].iterrows()]
            ln   = np.log(np.clip(vals, 1e-12, None))
            sig  = np.std(ln)
            med  = np.exp(np.mean(ln))
            medians.append(med)
            lo.append(np.exp(np.mean(ln) - sig))
            hi.append(np.exp(np.mean(ln) + sig))
        ax.plot(medians, IM_list_sorted, 'k', label='Median', zorder=3)
        ax.plot(lo,      IM_list_sorted, 'b', label='-σ',     zorder=3)
        ax.plot(hi,      IM_list_sorted, 'g', label='+σ',     zorder=3)

    ax.set_xlabel('Max drift ratio', fontdict={'family': 'Times New Roman', 'size': 12})
    ax.set_ylabel(ylabel,            fontdict={'family': 'Times New Roman', 'size': 12})
    ax.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 11})
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(FigName, dpi=600, format='jpg', bbox_inches='tight')
    plt.show()


# ── EDP 插值（单向） ───────────────────────────────────────────────────────────

def interp_edp_from_ida(
    ida_csv: Union[str, Path, pd.DataFrame],
    im_target: float,
    num_stories: int,
):
    """从单向 IDA 结果中提取目标 IM 处的 EDP 样本（各记录独立线性插值）。

    Returns
    -------
    tuple
        ``(drift_mat, accel_mat, res_arr, vel_mat, pgv_arr, extra_dict)``

        - **drift_mat** ``(n, N)`` – 最大层间位移角
        - **accel_mat** ``(n, N+1)`` – 峰值楼面加速度（g），首列为地面层
        - **res_arr**   ``(n,)``   – 残余层间位移角
        - **vel_mat**   ``(n, N+1)`` – 峰值楼面速度（m/s）
        - **pgv_arr**   ``(n,)`` or None – 峰值地面速度（m/s）
        - **extra_dict** ``dict`` – 用户自定义 EDP 插值结果
    """
    N = int(num_stories)
    df = ida_csv.copy() if isinstance(ida_csv, pd.DataFrame) else read_IDA_csv(ida_csv)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['Iffinish'] = df['Iffinish'].astype(bool)
    df = df.loc[df['Iffinish']].reset_index(drop=True)
    if df.empty:
        raise ValueError("没有收敛完成的记录，无法提取 EDP。")

    # 解析字符串列为数组
    for col in ['MaxDrift', 'MaxAbsAccel', 'MaxAbsVel', 'ResDrift']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: _parse_ida_array(v) if isinstance(v, str) else np.asarray(v, dtype=float))

    extra_cols = [c for c in df.columns if c not in _STANDARD_COLS_1D]
    for col in extra_cols:
        df[col] = df[col].apply(
            lambda v: _parse_ida_array(v) if isinstance(v, str) else np.asarray(v, dtype=float))

    drift_rows, accel_rows, res_rows, vel_rows, pgv_rows = [], [], [], [], []
    extra_rows = {c: [] for c in extra_cols}

    for _, rows in df.groupby('EQRecord', sort=False):
        drift = np.asarray(_interp_ida_value(rows, im_target, 'MaxDrift'), dtype=float)[:N]
        accel = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsAccel'), dtype=float)[:N]
        res   = _interp_ida_value(rows, im_target, 'ResDrift')
        res   = float(res) if not hasattr(res, '__len__') else float(np.max(res))

        drift_rows.append(drift)
        accel_rows.append(accel)
        res_rows.append(res)

        if 'MaxAbsVel' in df.columns:
            vel = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsVel'), dtype=float)
            vel_rows.append(vel[:N + 1] if len(vel) >= N + 1 else np.concatenate([[0.0], vel[:N]]))

        if 'PGV' in df.columns:
            pgv_rows.append(float(_interp_ida_value(rows, im_target, 'PGV')))

        for col in extra_cols:
            if col in rows.columns:
                extra_rows[col].append(np.asarray(_interp_ida_value(rows, im_target, col), dtype=float))

    drift_mat = np.clip(np.array(drift_rows, dtype=float), 1e-8, None)
    accel_raw = np.clip(np.array(accel_rows, dtype=float), 1e-8, None)
    accel_mat = np.hstack([np.full((len(accel_raw), 1), im_target * 0.4), accel_raw / 9800.0])
    res_arr   = np.clip(np.array(res_rows, dtype=float), 1e-8, None)
    vel_mat   = (np.clip(np.array(vel_rows, dtype=float) / 1000.0, 1e-8, None)
                 if vel_rows else np.full((len(drift_mat), N + 1), 10.0))
    pgv_arr   = np.clip(np.array(pgv_rows, dtype=float), 1e-8, None) if pgv_rows else None

    extra_dict = {}
    for col, vals in extra_rows.items():
        if vals:
            try:
                extra_dict[col] = np.array(vals, dtype=float)
            except (ValueError, TypeError):
                pass

    return drift_mat, accel_mat, res_arr, vel_mat, pgv_arr, extra_dict


# ── EDP 插值（双向） ───────────────────────────────────────────────────────────

def interp_edp_from_ida_bidir(
    ida_csv: Union[str, Path, pd.DataFrame],
    im_target: float,
    num_stories: int,
):
    """从双向 IDA 结果中提取目标 IM 处的 X/Y 双向 EDP 样本。

    Returns
    -------
    tuple
        ``(drift_X, drift_Y, accel_X, accel_Y, res_X, res_Y,
           vel_X, vel_Y, pgv_X, pgv_Y, extra_x_dict, extra_y_dict)``

        各矩阵形状：drift ``(n, N)``；accel/vel ``(n, N+1)``；res ``(n,)``。
    """
    N = int(num_stories)
    if isinstance(ida_csv, pd.DataFrame):
        df = ida_csv.copy()
        for col in ['MaxDrift_X', 'MaxDrift_Y', 'MaxAbsAccel_X', 'MaxAbsAccel_Y',
                    'MaxAbsVel_X', 'MaxAbsVel_Y']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: _parse_ida_array(v) if isinstance(v, str) else np.asarray(v, dtype=float))
    else:
        df = read_IDA_csv(ida_csv)

    df['Iffinish'] = df['Iffinish'].astype(bool)
    df = df.loc[df['Iffinish']].reset_index(drop=True)
    if df.empty:
        raise ValueError("双向 IDA 结果中没有收敛完成的记录。")

    has_vel = 'MaxAbsVel_X' in df.columns
    has_pgv = 'PGV_X' in df.columns

    extra_bases = [c[:-2] for c in df.columns
                   if c not in _STANDARD_COLS_2D and c.endswith('_X')]

    dX, dY, aX, aY, rX, rY, vX, vY, pX, pY = [], [], [], [], [], [], [], [], [], []
    eX = {b: [] for b in extra_bases}
    eY = {b: [] for b in extra_bases}

    df['_pair'] = df['EQRecord_X'].astype(str) + '|' + df['EQRecord_Y'].astype(str)
    for _, rows in df.groupby('_pair', sort=False):
        drift_x = np.asarray(_interp_ida_value(rows, im_target, 'MaxDrift_X'), dtype=float)[:N]
        drift_y = np.asarray(_interp_ida_value(rows, im_target, 'MaxDrift_Y'), dtype=float)[:N]

        ax_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsAccel_X'), dtype=float)[:N]
        ay_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsAccel_Y'), dtype=float)[:N]
        gnd    = np.array([im_target * 0.4])
        accel_x = np.concatenate([gnd, ax_raw / 9800.0])
        accel_y = np.concatenate([gnd, ay_raw / 9800.0])

        res_x = float(np.asarray(_interp_ida_value(rows, im_target, 'ResDrift_X'), dtype=float))
        res_y = float(np.asarray(_interp_ida_value(rows, im_target, 'ResDrift_Y'), dtype=float))

        if has_vel:
            vx_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsVel_X'), dtype=float)
            vy_raw = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsVel_Y'), dtype=float)
            vel_x = (vx_raw[:N + 1] if len(vx_raw) >= N + 1 else np.concatenate([[0.0], vx_raw[:N]])) / 1000.0
            vel_y = (vy_raw[:N + 1] if len(vy_raw) >= N + 1 else np.concatenate([[0.0], vy_raw[:N]])) / 1000.0
        else:
            vel_x = vel_y = np.full(N + 1, 10.0)

        dX.append(drift_x); dY.append(drift_y)
        aX.append(accel_x); aY.append(accel_y)
        rX.append(res_x);   rY.append(res_y)
        vX.append(vel_x);   vY.append(vel_y)
        if has_pgv:
            pX.append(float(_interp_ida_value(rows, im_target, 'PGV_X')))
            pY.append(float(_interp_ida_value(rows, im_target, 'PGV_Y')))
        for b in extra_bases:
            if b + '_X' in rows.columns:
                eX[b].append(np.asarray(_interp_ida_value(rows, im_target, b + '_X'), dtype=float))
            if b + '_Y' in rows.columns:
                eY[b].append(np.asarray(_interp_ida_value(rows, im_target, b + '_Y'), dtype=float))

    clip = lambda arr: np.clip(np.array(arr, dtype=float), 1e-8, None)
    drift_X, drift_Y = clip(dX), clip(dY)
    accel_X, accel_Y = clip(aX), clip(aY)
    res_X,   res_Y   = clip(rX), clip(rY)
    vel_X,   vel_Y   = clip(vX), clip(vY)
    pgv_X = clip(pX) if pX else None
    pgv_Y = clip(pY) if pY else None

    def _build_extra(src):
        out = {}
        for k, vals in src.items():
            if vals:
                try:
                    out[k] = np.array(vals, dtype=float)
                except (ValueError, TypeError):
                    pass
        return out

    return (drift_X, drift_Y, accel_X, accel_Y, res_X, res_Y,vel_X, vel_Y, pgv_X, pgv_Y, _build_extra(eX), _build_extra(eY))


# ── 双向结果转单向包络（供 Hazus 等单向模块使用） ─────────────────────────────

def IDA_bidir_to_envelope(ida_csv: Union[str, Path]) -> pd.DataFrame:
    """将双向 IDA 结果转换为 2D 等效包络（逐层取 X/Y 最大值）。

    输出格式与单向 IDA 兼容，可直接传入 Hazus 损失评估函数。
    """
    df = read_IDA_csv(ida_csv)

    def _env(a, b):
        a_ = np.asarray(a, dtype=float)
        b_ = np.asarray(b, dtype=float)
        if a_.size == 0: return b_
        if b_.size == 0: return a_
        n = min(len(a_), len(b_))
        return np.maximum(a_[:n], b_[:n])

    rows = []
    for _, row in df.iterrows():
        rows.append({
            'IM':          row['IM'],
            'EQRecord':    row['EQRecord_X'],
            'MaxDrift':    list(_env(row['MaxDrift_X'],    row['MaxDrift_Y'])),
            'MaxAbsAccel': list(_env(row['MaxAbsAccel_X'], row['MaxAbsAccel_Y'])),
            'MaxAbsVel':   list(_env(row.get('MaxAbsVel_X', np.array([])),
                                     row.get('MaxAbsVel_Y', np.array([])))),
            'ResDrift':    max(float(row['ResDrift_X']), float(row['ResDrift_Y'])),
            'Iffinish':    bool(row['Iffinish']),
            'tCurrent':    float(row.get('tCurrent_X', 0.0)),
            'TotalTime':   float(row.get('TotalTime', 0.0)),
        })
    return pd.DataFrame(rows)


# ── EDP 模拟（单向 IDA 结果） ─────────────────────────────────────────────────

def SimulateEDPGivenIM(
    IDA_result: pd.DataFrame, IM_list: list, N_Sim, betaM: float = 0
) -> pd.DataFrame:
    """基于单向 IDA 结果，在指定 IM 条件下蒙特卡洛模拟 EDP 样本。"""
    SimEDP = pd.DataFrame({'IM': [], 'MaxDrift': [], 'MaxAbsAccel': [], 'ResDrift': []})

    N_Sim_list = [N_Sim] * len(IM_list) if isinstance(N_Sim, int) else N_Sim
    c = Counter(IDA_result['IM'].values.tolist())
    IM_avail = [k for k, v in c.items() if v >= 3]
    if not IM_avail:
        return SimEDP

    sub = IDA_result[['IM', 'MaxDrift', 'MaxAbsAccel', 'ResDrift']].copy()
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            sub.iat[i, j] = np.array(sub.iloc[i, j]).max()

    means, covs = [], []
    for im in IM_avail:
        EDPs = sub.drop(columns=['IM'])[sub['IM'] == im].values.astype(float)
        _, m, cov, _, _, _ = IDAAnalysis.FEMACodeSimulatingEDP(EDPs, betaM, 10)
        means.append(m)
        covs.append(cov)

    for im, N in zip(IM_list, N_Sim_list):
        m   = IDAAnalysis.interpMatrix(math.log(im),  [math.log(i) for i in IM_avail], means)
        cov = IDAAnalysis.interpMatrix(math.log(im),  [math.log(i) for i in IM_avail], covs, True)
        N_real = max(N, 10)
        W, _, _, _ = IDAAnalysis.FEMACodeSimulatingEDPGivenlnMeanlncov(m, cov, betaM, N_real)
        W = W[:N, :]
        block = pd.DataFrame(
            np.concatenate([np.full((N, 1), im), W], axis=1),
            columns=['IM', 'MaxDrift', 'MaxAbsAccel', 'ResDrift'],
        )
        SimEDP = pd.concat([SimEDP, block], ignore_index=True)

    return SimEDP


# ── 高级接口封装类 ─────────────────────────────────────────────────────────────

class IDAAnalysis:
    """IDA 分析高级接口，统一支持单向与双向地震动。

    用法示例::

        from MDOFModel.analysis import IDA_2D

        # 单向分析
        ida = IDA_2D.IDAAnalysis(model)
        ida.Analyze(IM_list=[0.1, 0.3, 0.5, 1.0], NumPool=4)
        ida.SaveToCSV('IDA_1D.csv')

        # 双向分析（IM = Sa_gm）
        ida = IDA_2D.IDAAnalysis(model)
        records = IDA_2D.load_fema_records(bidir=True)
        ida.Analyze(IM_list=[0.1, 0.3, 0.5, 1.0], records=records, NumPool=4)
        ida.SaveToCSV('IDA_2D.csv')
    """

    def __init__(self, FEModel: IDAModelProtocol):
        self.FEModel     = FEModel
        self.IDA_result: pd.DataFrame = None

    @property
    def is_bidir(self) -> bool:
        """返回当前结果是否为双向 IDA。"""
        return self.IDA_result is not None and 'EQRecord_X' in self.IDA_result.columns

    def Analyze(
        self,
        IM_list: list,
        period: float = None,
        records: list = None,
        DeltaT='AsInRecord',
        NumPool: int = 1,
        ExtraEDP: dict = None,
        output_csv: Union[str, Path] = None,
        restart: bool = False,
    ) -> pd.DataFrame:
        """执行 IDA 分析并保存结果。

        Parameters
        ----------
        records : list or None
            ``None`` 使用 FEMA P-695 X 分量（单向）；
            ``list[str]`` 单向；``list[(str, str)]`` 双向。
        output_csv : str or Path, optional
            结果输出路径；每完成一条记录即写盘。
            若文件已存在且 ``restart=False``（默认），自动跳过已完成记录，
            实现无感断点续算；``restart=True`` 则忽略已有文件从头重算。
        restart : bool, default False
            ``True`` 时强制从头重算，忽略 ``output_csv`` 中的已有结果。
        """
        if period is None:
            period = float(self.FEModel.T1)
        self.IDA_result = IDA_f(self.FEModel, IM_list, period, records, DeltaT, NumPool, ExtraEDP, output_csv, restart)
        return self.IDA_result

    def SaveToCSV(self, csv_file: Union[str, Path]) -> None:
        """将分析结果保存为 CSV 文件。"""
        if self.IDA_result is None:
            raise RuntimeError("尚未执行分析，请先调用 Analyze()。")
        self.IDA_result.to_csv(Path(csv_file), index=False, encoding='utf-8-sig')

    @staticmethod
    def plot_IDA_results(
        IDA_result: Union[str, Path, pd.DataFrame],
        Stat: bool = True,
        FigName: Union[str, Path] = 'IDA.jpg',
    ):
        """绘制 IDA 曲线（自动识别单向/双向）。"""
        return plot_IDA(IDA_result, Stat=Stat, FigName=FigName)

    def SimulateEDPGivenIM(self, IM_list: list, N_Sim, betaM: float = 0) -> pd.DataFrame:
        """在指定 IM 级别下模拟 EDP 样本（仅适用于单向 IDA 结果）。"""
        return SimulateEDPGivenIM(self.IDA_result, IM_list, N_Sim, betaM)

    # ── 统计工具（用于 EDP 模拟） ──────────────────────────────────────────────

    @staticmethod
    def interpMatrix(x, xp: list, Yp: list, nonnegative: bool = False) -> np.ndarray:
        """在矩阵/向量列表中对目标标量 x 进行线性插值。"""
        if len(xp) == 1:
            xp = [0] + xp
            Yp = [0] + Yp
        inx = np.argsort(np.abs(x - np.array(xp)))
        Y = (Yp[inx[1]] - Yp[inx[0]]) * (x - xp[inx[0]]) / (xp[inx[1]] - xp[inx[0]]) + Yp[inx[0]]
        if nonnegative and np.any(Y <= 0):
            Y = Yp[inx[0]]
            if np.any(Y <= 0):
                Y = Yp[inx[1]]
        return Y

    @staticmethod
    def FEMACodeSimulatingEDP(EDPs: np.ndarray, betaM: float, num_realization):
        """由 EDP 样本估计对数正态参数并生成模拟样本。

        Returns
        -------
        tuple
            ``(W, lnEDPs_mean, lnEDPs_cov, R, ratio_mean, ratio_cov)``
        """
        EDPs = EDPs.astype(float)
        lnEDPs = np.log(EDPs)
        lnEDPs_mean = np.mean(lnEDPs, 0)[:, np.newaxis]
        lnEDPs_cov  = np.cov(np.transpose(lnEDPs))
        W, R, ratio_mean, ratio_cov = IDAAnalysis.FEMACodeSimulatingEDPGivenlnMeanlncov(
            lnEDPs_mean, lnEDPs_cov, betaM, num_realization)
        return W, lnEDPs_mean, lnEDPs_cov, R, ratio_mean, ratio_cov

    @staticmethod
    def FEMACodeSimulatingEDPGivenlnMeanlncov(
        lnEDPs_mean, lnEDPs_cov, betaM, num_realization
    ):
        """由对数空间均值和协方差矩阵生成 EDP 模拟样本（FEMA 方法）。

        Returns
        -------
        tuple
            ``(W, R, ratio_mean, ratio_cov)``，W 为线性空间模拟值。
        """
        num_var  = lnEDPs_cov.shape[1]
        rank     = np.linalg.matrix_rank(lnEDPs_cov)

        # 用认知不确定性 betaM 膨胀方差
        sigma   = np.sqrt(np.diag(lnEDPs_cov))[:, np.newaxis]
        R       = lnEDPs_cov / (sigma @ sigma.T)
        sigma2  = np.sqrt(sigma ** 2 + betaM ** 2)
        lnEDPs_cov_inf = R * (sigma2 @ sigma2.T)

        D2, L = np.linalg.eig(lnEDPs_cov_inf)
        idx   = D2.argsort()
        D2, L = D2[idx], L[:, idx]

        if rank < num_var:
            L  = L[:, num_var - rank:]
            D2 = D2[num_var - rank:]

        D2[D2 < 0] = 1e-6
        D_diag = np.diag(np.sqrt(D2))
        U      = np.random.normal(size=(rank if rank < num_var else num_var, num_realization))
        Z      = (L @ D_diag) @ U + lnEDPs_mean @ np.ones((1, num_realization))

        ratio_mean = np.mean(Z, 1) / lnEDPs_mean.T
        ratio_cov  = np.cov(Z) / lnEDPs_cov
        return np.exp(Z).T, R, ratio_mean, ratio_cov
