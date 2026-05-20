"""
Example 7 — 柱截面应变 ExtraEDP 回调（独立模块）

将回调函数定义在独立模块而非 ``__main__`` 中，使其可被 Python ``multiprocessing``
的 ``spawn`` 启动方式（Windows 默认）正确 pickle，从而支持 IDA 并行分析（NumPool > 1）。

并行安全说明
-----------
``extra_recorder_setup(tmp_dir)`` 中所有 recorder 输出文件均写入
``tmp_dir``（即 ``TmpDir/opensees_{UniqueRecorderPrefix}``），该目录由
``GeneralModelWrapper.DynamicAnalysis`` 按地震波文件名自动生成，
**在多进程中对每条地震波记录是唯一的**，因此不存在文件冲突。
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

# ── 各层代表性中柱参数（来自 Example_MRF_Model.py）──────────────────────────
# 中柱单元标签：col=3，story=1..6 → tag = 10000 + 100*story + 3
COL_ELEMENTS = [10103, 10203, 10303, 10403, 10503, 10603]

# 各层柱截面深度一半 d/2 (mm)：
#   story 1–2: W650 (d=650 mm) → 325 mm
#   story 3:   W550 (d=550 mm) → 275 mm
#   story 4–6: W450 (d=450 mm) → 225 mm
COL_HALF_D = [325.0, 325.0, 275.0, 225.0, 225.0, 225.0]


def setup_strain_recorders(tmp_dir: Path) -> None:
    """
    ``GeneralModelWrapper.extra_recorder_setup`` 回调。

    在每次动力分析的标准 recorder 注册完毕后调用，为各层代表性中柱（col=3）
    的底部截面（section 1）注册 EnvelopeElement 截面变形 recorder。

    EnvelopeElement deformation 输出格式：
        3 行（min / max / absMax） × 2 列（轴向应变 ε, 曲率 κ [rad/mm]）

    输出路径 ``tmp_dir / col_deform_{i}.out`` 保证多进程安全：
    ``tmp_dir = TmpDir/opensees_{UniqueRecorderPrefix}``，每条记录唯一。

    参数
    ----
    tmp_dir : Path
        本次分析的进程唯一临时目录（由框架传入）。
    """
    import openseespy.opensees as ops
    for i, ele_tag in enumerate(COL_ELEMENTS):
        ops.recorder(
            'EnvelopeElement',
            '-file', str(tmp_dir / f'col_deform_{i}.out'),
            '-ele',  ele_tag,
            'section', 1, 'deformation',
        )


def read_strain_results(model, tmp_dir: Path) -> None:
    """
    ``GeneralModelWrapper.extra_post_process`` 回调。

    在 ``ops.wipe()`` 和标准 ``_post_process`` 完成后调用，读取 EnvelopeElement
    截面变形包络文件，估算各层柱底极端纤维应变，并写入 ``model.MaxColStrain``。

    极端纤维应变（保守上界，弯轴叠加）：

        ε_extreme ≈ |ε_axial|_absMax + |κ|_absMax × (d/2)

    参数
    ----
    model : GeneralModelWrapper
        当前分析的模型实例；结果写入 ``model.MaxColStrain``（list[float]）。
    tmp_dir : Path
        本次分析的临时目录（与 setup_strain_recorders 使用同一个）。
    """
    strains = []
    for i, half_d in enumerate(COL_HALF_D):
        try:
            arr = np.atleast_2d(np.loadtxt(tmp_dir / f'col_deform_{i}.out'))
            if arr.shape[0] >= 3 and arr.shape[1] >= 2:
                strains.append(float(arr[2, 0]) + float(arr[2, 1]) * half_d)
            else:
                strains.append(float('nan'))
        except Exception:
            strains.append(float('nan'))
    model.MaxColStrain = strains
