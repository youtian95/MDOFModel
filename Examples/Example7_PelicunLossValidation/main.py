"""
Example 7 — Pelicun (FEMA P-58) 损失评估验证（使用已有 demands.csv + CMP_QNT.csv）

本例直接使用 H:\\CMP_PACTvsPelicun 中的需求文件和构件数量文件，
以当前版本的 pelicun 重新计算，并与原始参考结果进行对比。

与 Example 6 的区别：
  - Example 6 通过 NormQtyPact 自动生成非结构构件，从 IDA 结果提取 EDP
  - Example 7 直接使用已有的 demands.csv（11 条地震记录样本）和
    CMP_QNT.csv，调用 run_pelicun 进行验证

数据来源：H:\\CMP_PACTvsPelicun
  - demands.csv        : 3 层商业办公楼，11 条地震记录，含 PFA/PID/RID/SA
  - CMP_QNT.csv        : FEMA P-58 构件数量文件
  - input.json         : 原始 pelicun 配置（旧 API 格式，供参考）
  - DV_bldg_repair_agg.zip : 原始参考结果（旧版 pelicun 计算）

模型说明：
  - 3 层商业办公楼（OccupancyType: COM1）
  - 倒塌易损性：SA(T=1.13s)，中值 14.7 m/s²，β=0.4
  - 不可修复限值：RID 中值 0.01 rad，β=0.3
  - 替换费用：20,000,000 USD_2011
  - SampleSize：500
"""

import json
import shutil
import warnings
from pathlib import Path
import pandas as pd

# ── 源数据目录 ────────────────────────────────────────────────────────────────
SRC_DIR = Path(r'H:\CMP_PACTvsPelicun')

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent / 'Output'

# ── 配置参数（与 H:\CMP_PACTvsPelicun\input.json 一致）─────────────────────────
SAMPLE_SIZE       = 500
SEED              = 415
NUM_STORIES       = 3
OCCUPANCY_TYPE    = 'COM1'
REPLACEMENT_COST  = 20_000_000      # USD_2011

COLLAPSE_DEMAND_TYPE = 'SA_1.13'    # 谱加速度周期 1.13s
COLLAPSE_MEDIAN      = 14.7         # m/s²
COLLAPSE_LOGSTD      = 0.4

IRREPARABLE_MEDIAN   = 0.01         # rad
IRREPARABLE_LOGSTD   = 0.3


def _prepare_demand_csv(src: Path, dst: Path) -> str:
    """
    将 CMP_PACTvsPelicun/demands.csv 转换为当前 pelicun 格式：
      - 索引列从 'Row' 改为 'ID'
      - 去掉列名前缀 '1-'（pelicun 旧格式标记，当前版本不需要）
    """
    df = pd.read_csv(src, index_col=0)
    df.index.name = 'ID'
    df.columns = [c.removeprefix('1-') for c in df.columns]
    df.to_csv(dst)
    return str(dst)


if __name__ == '__main__':

    print('=' * 65)
    print('  Example 7: Pelicun 损失评估验证（CMP_PACTvsPelicun 数据）')
    print('=' * 65)
    print(f'  源数据目录  : {SRC_DIR}')
    print(f'  楼层数      : {NUM_STORIES}')
    print(f'  使用类型    : {OCCUPANCY_TYPE}')
    print(f'  替换费用    : {REPLACEMENT_COST:,} USD_2011')
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir   = OUT_DIR
    output_dir = work_dir / 'pelicun_output'
    output_dir.mkdir(exist_ok=True)

    # ── Step 1：准备输入文件 ─────────────────────────────────────────────────
    demand_csv = _prepare_demand_csv(
        SRC_DIR / 'demands.csv',
        work_dir / 'demands.csv',
    )

    cmp_dst = work_dir / 'CMP_QNT.csv'
    shutil.copy2(SRC_DIR / 'CMP_QNT.csv', cmp_dst)
    cmp_csv = str(cmp_dst)

    # ── Step 2：生成 DL_config.json（当前 pelicun API 格式）────────────────
    dl_config = {
        'GeneralInformation': {
            'units': {'force': 'N', 'length': 'm', 'temperature': 'C', 'time': 'sec'},
        },
        'DL': {
            'Demands': {
                'DemandFilePath': demand_csv,
                'Calibration': {
                    'ALL': {'DistributionFamily': 'lognormal'},
                    'PID': {
                        'DistributionFamily': 'lognormal',
                        'TruncateLower': '',
                        'TruncateUpper': '',
                    },
                },
            },
            'Asset': {
                'ComponentAssignmentFile': cmp_csv,
                'ComponentDatabase':       'FEMA P-58',
                'NumberOfStories':         str(NUM_STORIES),
                'OccupancyType':           OCCUPANCY_TYPE,
            },
            'Damage': {
                'DamageProcess': 'FEMA P-58',
                'CollapseFragility': {
                    'CapacityDistribution': 'lognormal',
                    'CapacityMedian':       COLLAPSE_MEDIAN,
                    'DemandType':           COLLAPSE_DEMAND_TYPE,
                    'Theta_1':              COLLAPSE_LOGSTD,
                },
                'IrreparableDamage': {
                    'DriftCapacityMedian': IRREPARABLE_MEDIAN,
                    'DriftCapacityLogStd': IRREPARABLE_LOGSTD,
                },
            },
            'Losses': {
                'Repair': {
                    'ConsequenceDatabase': 'FEMA P-58',
                    'MapApproach':         'Automatic',
                    'DecisionVariables':   {'Cost': True, 'Time': True},
                    'ReplacementCost': {
                        'Median': float(REPLACEMENT_COST),
                        'Unit':   'USD_2011',
                    },
                },
            },
            'Options': {
                'Seed':     SEED,
                'PrintLog': False,
                'LogFile':  str(work_dir / 'pelicun_log.txt'),
                'Sampling': {'SampleSize': SAMPLE_SIZE},
            },
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

    config_json = str(work_dir / 'DL_config.json')
    with open(config_json, 'w', encoding='utf-8') as fp:
        json.dump(dl_config, fp, indent=2)

    # ── Step 3：调用 run_pelicun ─────────────────────────────────────────────
    try:
        from pelicun.tools.DL_calculation import run_pelicun
    except ImportError as exc:
        raise ImportError('找不到 pelicun 包。请运行: pip install pelicun') from exc

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_pelicun(
            config_path      = config_json,
            demand_file      = demand_csv,
            output_path      = str(output_dir),
            realizations     = SAMPLE_SIZE,
            auto_script_path = None,
            custom_model_dir = None,
            output_format    = ['csv'],
            detailed_results = True,
            coupled_edp      = True,   # 直接使用 demands.csv 中的 11 条样本行
        )

    # ── Step 4：读取新计算结果 ───────────────────────────────────────────────
    agg_zip    = output_dir / 'DV_repair_agg.zip'
    agg_new    = pd.read_csv(agg_zip, index_col=0, compression='zip')

    cost_col = next((c for c in agg_new.columns
                     if 'cost' in c.lower() and 'parallel' not in c.lower()), None)
    time_col = next((c for c in agg_new.columns
                     if 'sequential' in c.lower()), None)

    print('─' * 65)
    print('  当前版本 pelicun 计算结果：')
    print('─' * 65)
    if cost_col:
        print(f'  平均修复费用      : {agg_new[cost_col].mean():>15,.0f} USD')
        print(f'  修复费用标准差    : {agg_new[cost_col].std():>15,.0f} USD')
        q = agg_new[cost_col].quantile([0.16, 0.50, 0.84])
        print(f'  费用 16/50/84 分位: '
              f'{q.iloc[0]:,.0f} / {q.iloc[1]:,.0f} / {q.iloc[2]:,.0f} USD')
    if time_col:
        print(f'  平均修复时间(seq) : {agg_new[time_col].mean():>15,.1f} 工人·天')

    # ── Step 5：与原始参考结果对比 ───────────────────────────────────────────
    ref_zip = SRC_DIR / 'DV_bldg_repair_agg.zip'
    if ref_zip.exists():
        agg_ref = pd.read_csv(ref_zip, index_col=0, compression='zip')
        ref_cost_col = next((c for c in agg_ref.columns
                             if 'cost' in c.lower()), None)
        ref_time_col = next((c for c in agg_ref.columns
                             if 'sequential' in c.lower()), None)
        print()
        print('─' * 65)
        print('  原始参考结果（旧版 pelicun，H:\\CMP_PACTvsPelicun）：')
        print('─' * 65)
        if ref_cost_col:
            print(f'  平均修复费用      : {agg_ref[ref_cost_col].mean():>15,.0f} USD')
            print(f'  修复费用标准差    : {agg_ref[ref_cost_col].std():>15,.0f} USD')
            q_ref = agg_ref[ref_cost_col].quantile([0.16, 0.50, 0.84])
            print(f'  费用 16/50/84 分位: '
                  f'{q_ref.iloc[0]:,.0f} / {q_ref.iloc[1]:,.0f} / {q_ref.iloc[2]:,.0f} USD')
        if ref_time_col:
            print(f'  平均修复时间(seq) : {agg_ref[ref_time_col].mean():>15,.1f} 工人·天')

    print()
    print(f'  输出目录 : {output_dir}')
    print('=' * 65)
