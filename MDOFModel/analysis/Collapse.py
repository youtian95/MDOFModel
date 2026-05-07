########################################################
# 基于 IDA 分析结果的倒塌筛选与分析模块。
########################################################

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# Hazus Table 5.9 中各设防等级 Complete 损伤状态层间位移角中值所在列索引（0-based）
_HAZUS_DESIGN_LEVEL_COL = {
    'high-code':     9,
    'moderate-code': 25,
    'low-code':      41,
    'pre-code':      57,
}

_HAZUS_TABLE_PATH = Path(__file__).resolve().parent.parent / 'Resources' / 'HazusData Table 5.9.csv'


def get_hazus_collapse_drift(
    building_type: str,
    design_level: str = 'moderate-code',
) -> float:
    """从 Hazus Table 5.9 查询指定结构类型和设防等级的 Complete 损伤状态层间位移角中值。

    该值通常作为倒塌判定的位移角阈值。

    Parameters
    ----------
    building_type : str
        Hazus 结构类型代码，如 S1L、C1M、W1 等。
    design_level : str, optional
        设防等级，可选 high-code、moderate-code、low-code、pre-code，
        默认 moderate-code。

    Returns
    -------
    float
        Complete 损伤状态层间位移角中值（如 0.08 表示 8%）。
    """
    if design_level not in _HAZUS_DESIGN_LEVEL_COL:
        raise ValueError(
            f"design_level 须为 {list(_HAZUS_DESIGN_LEVEL_COL.keys())} 之一，"
            f"当前值为 {design_level!r}。"
        )
    df = pd.read_csv(_HAZUS_TABLE_PATH, header=None)
    col_idx = _HAZUS_DESIGN_LEVEL_COL[design_level]
    match = df[df.iloc[:, 0] == building_type]
    if match.empty:
        raise ValueError(
            f"在 Hazus Table 5.9 中未找到结构类型 {building_type!r}。\n"
            f"可用类型：{df.iloc[4:, 0].tolist()}"
        )
    return float(match.iloc[0, col_idx])


def _parse_ida_array(value):
    if isinstance(value, str):
        return np.fromstring(value.strip().strip('[]').replace(',', ' '), sep=' ')
    return np.asarray(value, dtype=float)


def _max_drift_value(val) -> float:
    arr = _parse_ida_array(val)
    return float(arr.max()) if arr.size > 0 else 0.0


class CollapseAnalysis:
    """基于 IDA 结果的倒塌分析类。

    将 IDA CSV 路径和倒塌判定条件封装为对象，避免在每次调用
    filter_collapse / fit_collapse_fragility 时重复传入相同参数。

    Parameters
    ----------
    ida_csv : str or Path
        IDA 分析结果 CSV 文件路径（完整结果，含倒塌记录）。
    collapse_drift_limit : float or None, optional
        判定倒塌的最大层间位移角限值（绝对值，非百分比）。
        与 building_type 二选一；若同时提供，以此参数为准。
    building_type : str or None, optional
        Hazus 结构类型代码（如 S1L、C1M），自动从 Hazus Table 5.9 查询
        Complete 损伤状态层间位移角中值作为倒塌阈值。
    design_level : str, optional
        设防等级，可选 high-code、moderate-code、low-code、pre-code，
        默认 moderate-code。仅当 building_type 不为 None 时生效。

    Attributes
    ----------
    ida_csv : Path
        IDA 结果 CSV 路径。
    collapse_drift_limit : float or None
        实际使用的倒塌位移角阈值（若通过 building_type 查询则已换算为数值）。

    Examples
    --------
    >>> ca = CollapseAnalysis('IDA_results.csv', building_type='S1L')
    >>> result = ca.fit_collapse_fragility(fig_path='fragility.jpg')
    >>> filtered_df = ca.filter_collapse()
    """

    def __init__(
        self,
        ida_csv: Union[str, Path],
        collapse_drift_limit: Optional[float] = None,
        building_type: Optional[str] = None,
        design_level: str = 'moderate-code',
    ):
        self.ida_csv = Path(ida_csv)
        if collapse_drift_limit is not None:
            self.collapse_drift_limit = collapse_drift_limit
        elif building_type is not None:
            self.collapse_drift_limit = get_hazus_collapse_drift(building_type, design_level)
        else:
            self.collapse_drift_limit = None

    def filter_collapse(self) -> pd.DataFrame:
        """剔除倒塌记录，返回过滤后的 DataFrame。

        倒塌判定条件（满足其一即剔除）：

        1. Iffinish == False：分析未收敛。
        2. max(MaxDrift) >= collapse_drift_limit（仅当构造时提供了阈值时生效）。

        Returns
        -------
        pandas.DataFrame
            剔除倒塌记录后的 IDA 结果表格。
        """
        df = pd.read_csv(self.ida_csv)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        mask = df['Iffinish'].astype(bool)
        if self.collapse_drift_limit is not None:
            mask = mask & df['MaxDrift'].apply(_max_drift_value).lt(self.collapse_drift_limit)

        return df.loc[mask].reset_index(drop=True)

    def fit_collapse_fragility(
        self,
        fig_path: Union[str, Path, None] = None,
    ) -> dict:
        """使用最大似然估计（MLE）拟合对数正态倒塌易损性曲线。

        对数正态 CDF 模型：

            P(collapse | IM) = Phi( ln(IM / theta) / beta )

        其中 theta 为倒塌中值 Sa（g），beta 为对数标准差。

        Parameters
        ----------
        fig_path : str, Path, or None, optional
            若提供，则将倒塌易损性曲线保存至该路径（jpg 格式）。

        Returns
        -------
        dict
            'median' (float) — 倒塌易损性中值 Sa（g）；
            'logstd' (float) — 倒塌易损性对数标准差。
        """
        df = pd.read_csv(self.ida_csv)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['Iffinish'] = df['Iffinish'].astype(bool)

        drift_limit = self.collapse_drift_limit

        def _is_collapse(row) -> bool:
            if not row['Iffinish']:
                return True
            if drift_limit is not None:
                return _max_drift_value(row['MaxDrift']) >= drift_limit
            return False

        df['_collapse'] = df.apply(_is_collapse, axis=1)

        groups = df.groupby('IM')
        im_levels  = np.array(sorted(groups.groups.keys()), dtype=float)
        n_total    = np.array([len(groups.get_group(im)) for im in im_levels], dtype=float)
        n_collapse = np.array([groups.get_group(im)['_collapse'].sum() for im in im_levels], dtype=float)

        def neg_log_likelihood(params):
            ln_theta, beta = params
            if beta <= 0:
                return np.inf
            p = norm.cdf((np.log(im_levels) - ln_theta) / beta)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll = n_collapse * np.log(p) + (n_total - n_collapse) * np.log(1 - p)
            return -np.sum(ll)

        x0 = [np.mean(np.log(im_levels)), 0.4]
        result = minimize(neg_log_likelihood, x0, method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000})
        ln_theta, beta = result.x
        collapse_median = float(np.exp(ln_theta))
        collapse_logstd = float(abs(beta))

        if fig_path is not None:
            import matplotlib.pyplot as plt
            im_plot = np.linspace(im_levels.min() * 0.5, im_levels.max() * 1.5, 200)
            p_fit = norm.cdf((np.log(im_plot) - ln_theta) / collapse_logstd)
            fig, ax = plt.subplots()
            ax.plot(im_plot, p_fit, 'b-', label=f'MLE fit (θ={collapse_median:.3f}g, β={collapse_logstd:.3f})')
            ax.scatter(im_levels, n_collapse / n_total, color='red', zorder=5, label='Empirical')
            ax.set_xlabel('Sa (g)', fontdict={'family': 'Times New Roman', 'size': 12})
            ax.set_ylabel('P(Collapse)', fontdict={'family': 'Times New Roman', 'size': 12})
            ax.set_ylim(0, 1)
            ax.set_xlim(left=0)
            ax.legend(prop={'family': 'Times New Roman', 'size': 11})
            plt.savefig(fig_path, dpi=600, format='jpg', bbox_inches='tight')
            plt.close(fig)

        return {
            'median': collapse_median,
            'logstd': collapse_logstd,
        }
