########################################################
# record_utils.py – 地震动记录处理工具
#
# 提供从记录文件计算谱加速度（Sa）和峰值地面速度（PGV）的工具函数。
########################################################

import tempfile
from pathlib import Path

import eqsig.single
import numpy as np


def compute_sa(record_file: str, period: float) -> float:
    """计算单条地震动记录在给定周期处的弹性谱加速度（g）。

    读取 PEER AT2 或 TXT 格式地震动记录，利用 ``eqsig`` 计算
    单自由度弹性反应谱，返回给定周期处的谱加速度。

    Parameters
    ----------
    record_file : str
        地震动记录文件路径（不含扩展名），支持 ``.at2`` 和 ``.txt`` 格式。
    period : float
        计算谱加速度的目标周期（单位：s），通常取结构基本周期 T₁。

    Returns
    -------
    float
        谱加速度 Sa（g），阻尼比 5%。
    """
    from MDOFModel.analysis import ReadRecord  # 延迟导入，避免包级循环依赖

    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w', delete=False) as f:
        tmp = f.name
    try:
        dt, _ = ReadRecord.ReadRecord(record_file, tmp)
        accel = np.array(open(tmp).read().split(), dtype=float)
    finally:
        Path(tmp).unlink(missing_ok=True)
    sig = eqsig.single.AccSignal(accel * 9.8, dt)
    sig.generate_response_spectrum(response_times=np.array([period]))
    return float(sig.s_a[0] / 9.8)

