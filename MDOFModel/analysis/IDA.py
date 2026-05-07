########################################################
# 使用 Openseespy 并行执行 IDA（增量动力分析）。每次记录使用一个单独的进程。
# 
# 依赖库: 
# - openseespy, pandas, numpy, eqsig
########################################################

from collections import Counter
import copy
import multiprocessing as mp
import tempfile
import threading
import pandas as pd
from typing import Any, Protocol, Tuple, Union
import numpy as np
import eqsig.single
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

from ..models import MDOFOpenSees as mops
from . import ReadRecord

class IDAModelProtocol(Protocol):
    UniqueRecorderPrefix: str
    MaxDrift: list
    MaxAbsAccel: list
    MaxRelativeAccel: list
    MaxAbsVel: list
    ResDrift: list
    T1: float

    def DynamicAnalysis(self, record_file: str, scale_factor: float, ifprint: bool, delta_t) -> Tuple[bool, float, float]:
        ...

def IDA_1record(FEModel:IDAModelProtocol, IM_list:list, EQRecordfile:str, period:float, DeltaT = 'AsInRecord', _status_queue=None):
    """对单条地震动记录运行 IDA（增量动力分析）。

    Parameters
    ----------
    FEModel : IDAModelProtocol
        用于非线性动力分析的有限元模型。
    IM_list : list
        目标强度指标值（单位：g）。
    EQRecordfile : str
        地震动记录文件路径。
    period : float
        用于计算谱加速度的基本周期。
    DeltaT : str or float, optional
        动力分析的时间步长。使用 'AsInRecord' 以直接采用记录的步长。
    _status_queue : multiprocessing.Queue, optional
        多进程模式下传入的状态队列，用于向主进程上报 IM 进度与收敛信息。

    Returns
    -------
    pandas.DataFrame
        单条记录所有 IM 级别的 IDA 分析结果。
    """

    FEModel.UniqueRecorderPrefix = 'URP'+ Path(EQRecordfile).name +'_'
    record_name = Path(EQRecordfile).stem
    IDA_result = pd.DataFrame()

    # 计算谱加速度
    p = Path(EQRecordfile)
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w', delete=False) as _tmp_f:
        _tmp_path = _tmp_f.name
    try:
        dt, nPts = ReadRecord.ReadRecord(EQRecordfile, _tmp_path)
        with open(_tmp_path, 'r') as f:
            Allstr = f.read()
    finally:
        Path(_tmp_path).unlink(missing_ok=True)
    Allstr = Allstr.split()
    Accel = np.array(Allstr).astype(float)
    record = eqsig.AccSignal(Accel * 9.8, dt)
    record.generate_response_spectrum(response_times=np.array([period]))
    SA = record.s_a[0]/9.8

    # 单进程模式下显示 IM 内层进度条；多进程模式下通过队列上报
    n_im = len(IM_list)
    if _status_queue is None:
        im_iter = tqdm(enumerate(IM_list), total=n_im, desc=f"  {record_name[:28]}", leave=False, unit='IM')
    else:
        im_iter = enumerate(IM_list)

    for im_idx, IM in im_iter:
        # 多进程模式：分析开始前先上报 im_idx-1（当前进度），让主进程能立即创建子进度条
        if _status_queue is not None and im_idx == 0:
            _status_queue.put({
                'record': record_name,
                'im_idx': 0,
                'im_total': n_im,
                'IM': IM,
                'finished': True,
                'tCurrent': 0.0,
                'TotalTime': 0.0,
            })

        Iffinish, tCurrent, TotalTime = FEModel.DynamicAnalysis(
            str(Path(EQRecordfile)), IM/SA, False, DeltaT)

        if _status_queue is not None:
            _status_queue.put({
                'record': record_name,
                'im_idx': im_idx + 1,
                'im_total': n_im,
                'IM': IM,
                'finished': bool(Iffinish),
                'tCurrent': tCurrent,
                'TotalTime': TotalTime,
            })
        else:
            postfix = f"{'FAIL' if not Iffinish else 'OK'} t={tCurrent:.1f}/{TotalTime:.1f}s"
            im_iter.set_postfix_str(postfix)
            if not Iffinish:
                tqdm.write(
                    f"  [{record_name}] IM={IM:.3f}g: FAILED, "
                    f"t={tCurrent:.2f}/{TotalTime:.2f}s"
                )

        data = {'IM':IM,'EQRecord':EQRecordfile,'MaxDrift':[FEModel.MaxDrift],
            'MaxAbsAccel':[FEModel.MaxAbsAccel],'MaxRelativeAccel':[FEModel.MaxRelativeAccel],
            'MaxAbsVel':[getattr(FEModel, 'MaxAbsVel', [])],
            'ResDrift':FEModel.ResDrift,'Iffinish':bool(Iffinish), 'tCurrent':tCurrent, 'TotalTime':TotalTime}
        IDA_result=pd.concat([IDA_result,pd.DataFrame(data)], ignore_index=True)

    return IDA_result

def IDA_f(FEModel:IDAModelProtocol, IM_list:list, period:float, EQRecordfile_list:list=None, DeltaT = 'AsInRecord', NumPool = 1):
    """可并行对多条记录进行 IDA 分析。

    Parameters
    ----------
    FEModel : IDAModelProtocol
        基础有限元模型。
    IM_list : list
        目标强度指标值（单位：g）。
    period : float
        用于计算谱加速度的基本周期。
    EQRecordfile_list : list, optional
        地震动记录文件路径列表。默认使用 FEMA P-695 远场地震动记录。
    DeltaT : str or float, optional
        动力分析的时间步长。
    NumPool : int, optional
        工作进程数。设为 1 时串行执行。

    Returns
    -------
    pandas.DataFrame
        所有记录的组合 IDA 结果。
    """

    if EQRecordfile_list is None:
        from .. import __file__ as mdof_file
        fema_dir = Path(mdof_file).parent / 'Resources' / 'FEMA_P-695_far-field_ground_motions'
        meta_path = fema_dir / 'MetaData.txt'
        if meta_path.exists():
            T = pd.read_table(str(meta_path), sep=',')
            EQRecordfile_list = [str(fema_dir / str.replace(x, '.txt', '')) for x in T['AccelXfile'].to_list()]
        else:
            raise FileNotFoundError(f"Default FEMA P-695 records metadata not found at {meta_path}.")

    IDA_result = pd.DataFrame({'IM':[],'EQRecord':[],
        'MaxDrift':[],'MaxAbsAccel':[],'MaxRelativeAccel':[],'ResDrift':[],'Iffinish':[]})

    total = len(EQRecordfile_list)

    if NumPool == 1:
        for EQRecordfile in tqdm(EQRecordfile_list, desc='IDA', unit='record', total=total):
            FEModel_ = copy.deepcopy(FEModel)
            IDA_1RecordResult = IDA_1record(FEModel_,IM_list,EQRecordfile,period,DeltaT)
            IDA_result=pd.concat([IDA_result,IDA_1RecordResult], ignore_index=True)
    else:
        with mp.Manager() as manager:
            status_queue = manager.Queue()
            stop_event = threading.Event()

            def _progress_display(queue, stop_event):
                # 为每个并发进程分配一个子进度条槽位 (position 1..NumPool)
                available_positions = list(range(1, NumPool + 1))
                record_bars = {}  # record_name -> (tqdm_bar, position)

                while not stop_event.is_set() or not queue.empty():
                    try:
                        msg = queue.get(timeout=0.2)
                        record = msg['record']
                        im_idx  = msg['im_idx']
                        im_total = msg['im_total']

                        # 首次出现：分配一个槽位并创建进度条
                        if record not in record_bars:
                            pos = available_positions.pop(0) if available_positions else NumPool
                            bar = tqdm(total=im_total,
                                       desc=f"  {record[:25]}",
                                       position=pos, leave=False, unit='IM',
                                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
                            record_bars[record] = (bar, pos)

                        bar, pos = record_bars[record]
                        bar.n = im_idx
                        bar.refresh()

                        # 该记录所有 IM 分析完毕：关闭进度条并释放槽位
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
                target=_progress_display, args=(status_queue, stop_event), daemon=True)
            disp_thread.start()

            with mp.Pool(NumPool) as pool:
                futures = [
                    pool.apply_async(IDA_1record,
                        args=(copy.deepcopy(FEModel), IM_list, EQRecordfile, period, DeltaT, status_queue))
                    for EQRecordfile in EQRecordfile_list
                ]
                with tqdm(total=total, desc='IDA', unit='record', position=0) as pbar:
                    for future in futures:
                        IDA_result = pd.concat([IDA_result, future.get()], ignore_index=True)
                        pbar.update(1)

            stop_event.set()
            disp_thread.join(timeout=3.0)

    return IDA_result

def SimulateEDPGivenIM(IDA_result:pd.DataFrame, IM_list:list, N_Sim, betaM:float = 0) -> pd.DataFrame:
    """基于 IDA 结果中对应 IM 条件下生成 EDP 样本。

    Parameters
    ----------
    IDA_result : pandas.DataFrame
        包含 IM 和 EDP 响应列的 IDA 输出表格。
    IM_list : list
        需要生成 EDP 样本的 IM 目标值。
    N_Sim : int or list
        每个 IM 的模拟次数。如果是整数，则所有 IM 级别使用相同次数。
    betaM : float, optional
        认知不确定性参数，用于放大协方差。

    Returns
    -------
    pandas.DataFrame
        被模拟生成的具有 IM、MaxDrift、MaxAbsAccel 与 ResDrift 列的 EDP 表格。
    """

    SimEDP = pd.DataFrame({'IM':[],'MaxDrift':[],'MaxAbsAccel':[],'ResDrift':[]})

    if isinstance(N_Sim,int):
        N_Sim = [N_Sim]*len(IM_list)

    # 删除那些地震记录数量少于3条的 IM 级别
    c = Counter(IDA_result['IM'].values.tolist())
    for keys in list(c.keys()):
        if c[keys] < 3:
            del(c[keys]) 
    IM_list_original = list(c.keys())
    # IM_list_original = [0]+IM_list_original

    if len(IM_list_original) == 0:
        return SimEDP

    # max EDP
    IDA_result = IDA_result[['IM','MaxDrift','MaxAbsAccel','ResDrift']]
    for i in range(0,  IDA_result.shape[0]):
        for j in range(0, IDA_result.shape[1]):
            IDA_result.iat[i,j] = np.array(IDA_result.iloc[i,j]).max()
    # newdf = pd.DataFrame([[0]*4], columns=list(IDA_result.columns))
    # IDA_result = pd.concat([newdf,IDA_result], ignore_index=True)

    # 提取 EDP 在对数空间下的原始均值和对数标准差
    lnEDPs_mean_list_original = []
    lnEDPs_cov_list_original = []
    for IM in IM_list_original:
        EDPs = IDA_result.drop(columns=['IM'])[IDA_result['IM']==IM].values
        _,lnEDPs_mean,lnEDPs_cov,_,_,_ = IDAAnalysis.FEMACodeSimulatingEDP(EDPs, betaM, 10)
        lnEDPs_mean_list_original.append(lnEDPs_mean)
        lnEDPs_cov_list_original.append(lnEDPs_cov)

    # 模拟 EDP
    assert len(IM_list)==len(N_Sim)
    for IM,N in zip(IM_list,N_Sim):
        lnEDPs_mean = IDAAnalysis.interpMatrix(math.log(IM), [math.log(im) for im in IM_list_original], lnEDPs_mean_list_original)
        lnEDPs_cov = IDAAnalysis.interpMatrix(math.log(IM), [math.log(im) for im in IM_list_original], lnEDPs_cov_list_original, True)
        if N<10:
            N_real = 10
        else:
            N_real = N
        W,_,_,_ = IDAAnalysis.FEMACodeSimulatingEDPGivenlnMeanlncov(lnEDPs_mean,lnEDPs_cov,betaM,N_real)
        W = W[0:N,:]
        newdf = pd.DataFrame(np.concatenate((np.array([[IM]]*N),W),axis=1), columns=list(SimEDP.columns))
        SimEDP = pd.concat([SimEDP,newdf], ignore_index=True)

    return SimEDP


def _parse_ida_array(value):
    if isinstance(value, str):
        return np.fromstring(value.strip().strip('[]').replace(',', ' '), sep=' ')
    return np.asarray(value, dtype=float)


def _interp_ida_value(rows: pd.DataFrame, im_target: float, column: str):
    im_values = rows['IM'].astype(float).to_numpy()
    order = np.argsort(im_values)
    rows = rows.iloc[order].reset_index(drop=True)
    im_values = im_values[order]

    exact = np.where(np.isclose(im_values, im_target))[0]
    if len(exact) > 0:
        return rows.iloc[exact[0]][column]

    if im_target <= im_values[0]:
        return rows.iloc[0][column]
    if im_target >= im_values[-1]:
        return rows.iloc[-1][column]

    upper_idx = int(np.searchsorted(im_values, im_target, side='right'))
    lower_idx = upper_idx - 1
    im_low = im_values[lower_idx]
    im_high = im_values[upper_idx]
    weight = (im_target - im_low) / (im_high - im_low)

    low = _parse_ida_array(rows.iloc[lower_idx][column])
    high = _parse_ida_array(rows.iloc[upper_idx][column])
    if low.shape != high.shape:
        raise ValueError(
            f"Cannot interpolate IDA column '{column}' because record "
            f"{rows.iloc[lower_idx]['EQRecord']} has inconsistent array sizes."
        )

    return low + weight * (high - low)


def interp_edp_from_ida(
    ida_csv: Union[str, Path, 'pd.DataFrame'],
    im_target: float,
    num_stories: int,
):
    """
    Extract EDP samples from IDA results at a target IM using per-record
    linear interpolation between adjacent IM levels.
    """
    N = int(num_stories)
    if isinstance(ida_csv, pd.DataFrame):
        df = ida_csv.copy()
    else:
        df = pd.read_csv(Path(ida_csv))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['Iffinish'] = df['Iffinish'].astype(bool)
    df = df.loc[df['Iffinish'], :].reset_index(drop=True)

    if df.empty:
        raise ValueError("No finished IDA records are available for EDP interpolation.")

    for col in ('MaxDrift', 'MaxAbsAccel', 'MaxAbsVel', 'ResDrift'):
        if col not in df.columns:
            continue
        for idx in df.index:
            val = df.at[idx, col]
            if isinstance(val, str):
                arr = _parse_ida_array(val)
                df.at[idx, col] = arr if arr.size != 1 else float(arr[0])

    drift_rows = []
    accel_rows = []
    res_rows = []
    vel_rows = []

    for _, rows in df.groupby('EQRecord', sort=False):
        if rows.empty:
            continue

        drift = np.asarray(_interp_ida_value(rows, im_target, 'MaxDrift'), dtype=float)[:N]
        accel = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsAccel'), dtype=float)[:N]
        res = _interp_ida_value(rows, im_target, 'ResDrift')
        res = float(res) if not hasattr(res, '__len__') else float(np.max(res))

        drift_rows.append(drift)
        accel_rows.append(accel)
        res_rows.append(res)

        if 'MaxAbsVel' in rows.columns:
            vel = np.asarray(_interp_ida_value(rows, im_target, 'MaxAbsVel'), dtype=float)
            if len(vel) >= N + 1:
                vel_rows.append(vel[:N + 1])
            else:
                vel_rows.append(np.concatenate([np.array([0.0]), vel[:N]]))

    drift_mat = np.clip(np.array(drift_rows, dtype=float), 1e-8, None)

    accel_raw = np.clip(np.array(accel_rows, dtype=float), 1e-8, None)
    accel_g = accel_raw / 9800.0
    ground = np.full((len(accel_g), 1), im_target * 0.4)
    accel_mat = np.hstack([ground, accel_g])

    res_arr = np.clip(np.array(res_rows, dtype=float), 1e-8, None)

    if 'MaxAbsVel' in df.columns and vel_rows:
        vel_mat = np.clip(np.array(vel_rows, dtype=float) / 1000.0, 1e-8, None)
    else:
        vel_mat = np.full((len(drift_mat), N + 1), 10.0)

    return drift_mat, accel_mat, res_arr, vel_mat


def read_IDA_results_csv(csv_file: Union[str, Path]) -> pd.DataFrame:
    """读取 IDA CSV 并将字符串形式的数组列还原为 numpy 数组。"""
    def _parse_array_text(value):
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text == '' or text.lower() == 'nan':
            return np.array([])
        return np.fromstring(text.strip('[]').replace(',', ' '), sep=' ')

    converters = {
        col: _parse_array_text
        for col in ['MaxDrift', 'MaxAbsAccel', 'MaxRelativeAccel', 'MaxAbsVel', 'ResDrift']
    }
    df = pd.read_csv(Path(csv_file), converters=converters)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def plot_IDA_results_from_csv(
    csv_file: Union[str, Path],
    Stat: bool = True,
    FigName: Union[str, Path] = 'IDA.jpg',
):
    """直接从 IDA 结果 CSV 文件绘制 IDA 曲线图。"""
    IDA_result = read_IDA_results_csv(csv_file)
    return IDAAnalysis.plot_IDA_results(IDA_result, Stat=Stat, FigName=FigName)


class IDAAnalysis():
    """用于执行 IDA 及其 EDP 模拟的高级接口封装类。"""

    FEModel:IDAModelProtocol = None
    IDA_result:pd.DataFrame = None

    def __init__(self, FEModel:IDAModelProtocol):
        """使用给定的有限元模型初始化 IDA 包装器。

        Parameters
        ----------
        FEModel : IDAModelProtocol
            在分析中使用的有限元模型对象。
        """
        self.FEModel = FEModel

    def Analyze(self, IM_list:list, period:float=None, EQRecordfile_list:list=None, DeltaT = 'AsInRecord', NumPool = 1) -> pd.DataFrame:
        """执行当前模型的 IDA 分析并保存结果。

        Parameters
        ----------
        IM_list : list
            目标强度指标值（单位：g）。
        period : float, optional
            用于计算谱加速度的基本周期。若为 None，将尝试从 FEModel 的 'T1' 或 'FundamentalPeriod' 属性读取。
        EQRecordfile_list : list, optional
            地震动记录文件路径列表。默认使用 FEMA P-695 远场地震动记录。
        DeltaT : str or float, optional
            动力分析的时间步长。
        NumPool : int, optional
            用于并行处理的进程数。

        Returns
        -------
        pandas.DataFrame
            所有对应记录和 IM 级别的 IDA 分析结果。
        """
        if period is None:
            if hasattr(self.FEModel, 'T1'):
                period = float(self.FEModel.T1)
            else:
                raise ValueError("Model must have a 'T1' property when period is not provided.")

        self.IDA_result = IDA_f(self.FEModel, IM_list, period, EQRecordfile_list, DeltaT, NumPool)
        return self.IDA_result

    def SaveToCSV(self, OutputCSVFile: Union[str, Path]) -> None:
        """将 IDA 分析结果保存为 CSV 文件。

        输出格式与 ``Tool_IDA.py`` 中的保存方式完全一致。

        Parameters
        ----------
        OutputCSVFile : str or Path
            输出 CSV 文件路径。
        """
        if self.IDA_result is None:
            raise RuntimeError("尚未执行 IDA 分析，请先调用 Analyze() 方法。")
        self.IDA_result.to_csv(Path(OutputCSVFile), index=False, encoding='utf-8-sig')

    def plot_IDA_results(IDA_result, Stat:bool = False, FigName:Union[str, Path] = 'IDA.jpg'):
        """根据结果绘制单条曲线或带有统计信息的 IDA 包络图。

        Parameters
        ----------
        IDA_result : str or pandas.DataFrame
            可以是 CSV 文件的路径或已读取到内存的 IDA 特征表格 DataFrame。
        Stat : bool, optional
            如果为 False，则绘制每条特定记录的 IDA 曲线；如果为 True，则绘制中位数和正负一倍标准差包络线。
        FigName : str, optional
            输出图片的文件名。
        """
        if isinstance(IDA_result, (str, Path)):
            IDA_result = read_IDA_results_csv(IDA_result)
        elif isinstance(IDA_result, pd.DataFrame):
            pass
        else:
            raise ValueError('IDA_result should be a file name or a pandas DataFrame')

        cm = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots()   # figsize=(8*cm, 6*cm)
        EQRecordFile_list = list(Counter(IDA_result['EQRecord'].values).keys())
        if not Stat:
            for EQRecordFile in EQRecordFile_list:
                rows = IDA_result[IDA_result['EQRecord'] == EQRecordFile].sort_values('IM')
                ax.plot([max(drlist) for drlist in rows['MaxDrift'].values], rows['IM'].values)
        else:
            for i, EQRecordFile in enumerate(EQRecordFile_list):
                rows = IDA_result[IDA_result['EQRecord'] == EQRecordFile].sort_values('IM')
                ax.plot(
                    [max(drlist) for drlist in rows['MaxDrift'].values],
                    rows['IM'].values,
                    color='0.75',
                    linewidth=0.6,
                    alpha=0.8,
                    label='Records' if i == 0 else None,
                    zorder=1,
                )

            IM_list = sorted(Counter(list(IDA_result['IM'].values)).keys())
            EDPmax_median = []
            EDPmax_1sigma_minus = []
            EDPmax_1sigma_plus = []
            for im in IM_list:
                EDP_values = [np.array(drlist).max() for drlist in 
                    IDA_result['MaxDrift'][IDA_result['IM']==im].values]
                EDPmax_median.append(np.exp(np.mean(np.log(EDP_values))))
                EDPmax_1sigma_minus.append(np.exp(np.log(EDPmax_median[-1]) - np.std(np.log(EDP_values))))
                EDPmax_1sigma_plus.append(np.exp(np.log(EDPmax_median[-1]) + np.std(np.log(EDP_values))))
            ax.plot(EDPmax_median,IM_list,'k',label='Median', zorder=3)
            ax.plot(EDPmax_1sigma_minus,IM_list,'b',label='-sigma', zorder=3)
            ax.plot(EDPmax_1sigma_plus,IM_list,'g',label='+sigma', zorder=3)

        plt.xticks(fontproperties = 'Times New Roman', fontsize=12)
        plt.yticks(np.arange(0, 2, 0.2), fontproperties = 'Times New Roman', fontsize=12)
        # 指定横纵坐标的字体以及字体大小，记住是fontsize不是size。yticks上我还用numpy指定了坐标轴的变化范围。

        plt.legend(loc='lower right', prop={'family':'Times New Roman', 'size':12})
        # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混

        # plt.title('1000 samples', fontdict={'family' : 'Times New Roman', 'size':12})
        # 指定图上标题的字体及大小

        plt.xlabel('Drift ratio', fontdict={'family' : 'Times New Roman', 'size':12})
        plt.ylabel('Spectral accelerations (g)', fontdict={'family' : 'Times New Roman', 'size':12})
        # 指定横纵坐标描述的字体及大小

        plt.savefig(FigName, dpi=600, format='jpg', bbox_inches="tight")
        # 保存文件，dpi指定保存文件的分辨率
        # bbox_inches="tight" 可以保存图上所有的信息，不会出现横纵坐标轴的描述存掉了的情况

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(IM_list))

        plt.show()

    @staticmethod
    def plot_IDA_results_from_csv(
        csv_file: Union[str, Path],
        Stat: bool = True,
        FigName: Union[str, Path] = 'IDA.jpg',
    ):
        """直接从 IDA 结果 CSV 文件绘制 IDA 曲线图。"""
        return plot_IDA_results_from_csv(csv_file, Stat=Stat, FigName=FigName)

    def SimulateEDPGivenIM(self, IM_list:list, N_Sim, betaM:float = 0) -> pd.DataFrame:
        """利用已存储的 IDA 结果，在指定的 IM 级别下模拟大量 EDP 值。

        Parameters
        ----------
        IM_list : list
            需要进行 EDP 样本采样的 IM 值列表。
        N_Sim : int or list
            在每个 IM 级别上需要模拟的样本数量。
        betaM : float, optional
            认知可变性（认识不确定性）参数。

        Returns
        -------
        pandas.DataFrame
            生成的 EDP 模拟样本。
        """

        SimEDP = SimulateEDPGivenIM(self.IDA_result,IM_list,N_Sim,betaM)

        return SimEDP

    def interpMatrix(x,xp:list,Yp:list, nonnegative:bool = False)->np.array:
        """在一组矩阵/向量列表中，插值得到附近两点之间的目标数组。

        Parameters
        ----------
        x : float
            用于插值的目标标量。
        xp : list
            已知的标量样本点。
        Yp : list
            在已知采样点处的对应值。通常每个元素是一个 numpy 数组（向量或矩阵）。
        nonnegative : bool, optional
            如果为 True，并在插值过程中产生非正数条目时，退回使用最近的一个有效样本值。

        Returns
        -------
        numpy.ndarray
            插值计算得到的结果。
        """
        # x: 标量
        # xp: list[float]
        # Yp: list[np.array]

        if len(xp)==1:
            xp = [0]+xp
            Yp = [0]+Yp

        inx = np.argsort(np.abs(x-np.array(xp)))

        Y = (Yp[inx[1]]-Yp[inx[0]])*(x-xp[inx[0]])/(xp[inx[1]]-xp[inx[0]]) + Yp[inx[0]]

        if nonnegative and (np.sum(Y<=0)>0):
            Y = Yp[inx[0]]
            if np.sum(Y<=0)>0:
                Y = Yp[inx[1]]
                
        return Y

    def FEMACodeSimulatingEDPGivenlnMeanlncov(lnEDPs_mean,lnEDPs_cov,betaM,num_realization):
        """依靠对数空间的均值和方差生成大量的 EDP 样本实现。

        Parameters
        ----------
        lnEDPs_mean : numpy.ndarray
            对数 EDP 的均值向量，形状为 (N_var, 1)。
        lnEDPs_cov : numpy.ndarray
            对数 EDP 的协方差矩阵，形状为 (N_var, N_var)。
        betaM : float
            认知不确定参数，在此用于人为放大方差。
        num_realization : int
            模拟实现的数量。

        Returns
        -------
        tuple
            (W, R, ratio_mean, ratio_cov)，其中 W 表示在线性空间的 EDP 模拟值。
        """

        num_var = lnEDPs_cov.shape[1]

        # 查找并在 lnEDPs_cov_rank 变量中保存协方差矩阵的秩
        lnEDPs_cov_rank=np.linalg.matrix_rank(lnEDPs_cov)
        # 利用认知变异性膨胀方差
        sigma = np.sqrt(np.diag(lnEDPs_cov))[:,np.newaxis] # 先开平方避免下溢出或上溢
        sigmap2 = sigma * sigma
        R = lnEDPs_cov / (sigma @ (sigma.transpose())) 
        sigmap2 = sigmap2 + betaM**2    # Inflating variance for β m
        sigma=np.sqrt(sigmap2)
        sigma2 = sigma @ (sigma.T)
        lnEDPs_cov_inflated=R*sigma2

        # 寻找协方差阵的特征值及其对应的特征向量（即 D2_total 与 L_total）
        D2_total,L_total = np.linalg.eig(lnEDPs_cov_inflated)
        idx = D2_total.argsort()
        D2_total = D2_total[idx]
        L_total = L_total[:,idx]
        
        # 将 L_total 划分为对应正特征值的 L_use
        if lnEDPs_cov_rank >= num_var:
            L_use =L_total
        else:
            L_use = L_total[:, (num_var- lnEDPs_cov_rank):]
            # 因为L_use为特征值从小到大排列，所以0特征值在前面
            
        # 将 D2_total 中对应于正特征值的部分提取为 D2_use
        if lnEDPs_cov_rank >= num_var:
            D2_use = D2_total
        else:
            D2_use = D2_total[num_var- lnEDPs_cov_rank:]
        
        # 求 D2_use 的平方根，记为 D_use
        # 创建对角矩阵
        # 如果有任何负数，取为10**(-6)
        D2_use[D2_use<0] = 10**(-6)
        D_use = np.diag(np.power(D2_use, 0.5))

        # 生成标准正态随机数
        if lnEDPs_cov_rank >= num_var:
            U = np.random.normal(size=(num_realization, num_var))
        else:
            U = np.random.normal(size=(num_realization, lnEDPs_cov_rank))
            
        U = U.T

        # 创建 Lambda = D_use . L_use
        Lambda = L_use @ D_use
        # 创建最终的实现结果矩阵 
        Z = Lambda @ U + lnEDPs_mean @ np.ones((1,num_realization))
        lnEDPs_sim_mean = np.mean(Z,1)  # 行向量
        lnEDPs_sim_cov = np.cov(Z)
        ratio_mean = lnEDPs_sim_mean / (lnEDPs_mean.T)
        ratio_cov = lnEDPs_sim_cov / lnEDPs_cov
        W = np.exp(Z).T

        return W,R,ratio_mean,ratio_cov

    def FEMACodeSimulatingEDP(EDPs:np.array, betaM:float, num_realization):
        """根据 EDP 数据估计对数正态参数并模拟 EDP 样本。

        Parameters
        ----------
        EDPs : numpy.ndarray
            形状为 (N_sample, N_var) 的原始 EDP 数据样本。
        betaM : float
            认知不确定性参数。
        num_realization : int
            模拟实现的数量。

        Returns
        -------
        tuple
            (W, lnEDPs_mean, lnEDPs_cov, R, ratio_mean, ratio_cov).
        """
        # 返回:
        #   W:  N_sim x N_var
        #
        # 使用示例:
        # W,lnEDPs_mean,R,ratio_mean,ratio_cov = FEMACodeSimulatingEDP(
        #     np.array([[1,2,4],[0.1,0.2,0.5],[8,9,10],[5,2,1]]),0.3,1000)

        EDPs = EDPs.astype(float)

        # 对原 EDP 进行自然对数变换形成 lnEDPs
        lnEDPs = np.log(EDPs)
        num_var = lnEDPs.shape[1]

        # 寻找 lnEDPs 的对数空间均值
        lnEDPs_mean = np.mean(lnEDPs,0)[:,np.newaxis]

        # 计算该对数组的协方差阵
        lnEDPs_cov = np.cov(np.transpose(lnEDPs))

        W,R,ratio_mean,ratio_cov = IDAAnalysis.FEMACodeSimulatingEDPGivenlnMeanlncov(
            lnEDPs_mean,lnEDPs_cov,betaM,num_realization)
        
        return W,lnEDPs_mean,lnEDPs_cov,R,ratio_mean,ratio_cov


