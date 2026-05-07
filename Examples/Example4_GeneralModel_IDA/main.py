from pathlib import Path
import time 
import sys
import pandas as pd
import numpy as np
import openseespy.opensees as ops

examples_dir = str(Path(__file__).resolve().parent.parent)
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

from MDOFModel.analysis import IDA
from MDOFModel.analysis import Collapse
from MDOFModel.models.GeneralModelWrapper import GeneralModelWrapper

from Example_MRF_Model import build_model

floor_nodes = [103, 203, 303, 403, 503, 603]             
story_heights = [5000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0]
base_nodes = [1, 2, 3, 4, 5]

CFDir = Path(__file__).resolve().parent / "Output"
CFDir.mkdir(parents=True, exist_ok=True)

# 创建普通模型的 Wrapper 实例
wrapper_model = GeneralModelWrapper(
    build_model_func = build_model,
    floor_nodes = floor_nodes,
    story_heights = story_heights,
    dof = 1,                 
    base_nodes = base_nodes,
    g_factor = 9800.0,       # 记录波原始单位g乘以 9800.0 变成毫米 mm/s^2 
)

if __name__ == '__main__':

    print("====== 开始运行 GeneralModelWrapper 的测试分析 ======")

    # 生成由 0.1g 到 3.0g 的10点长的一系列IM作为IDA输入
    IM_list = np.linspace(0.1, 3.0, 10).tolist()
    
    IDA_obj = IDA.IDAAnalysis(wrapper_model)
    IDA_result = IDA_obj.Analyze(IM_list, NumPool=8)

    # 保存完整 IDA 结果（含倒塌记录，供倒塌分析使用）
    IDA_obj.SaveToCSV(CFDir/'IDA_results.csv')

    # 倒塌分析：MLE 拟合倒塌易损性 + 筛选非倒塌记录
    ca = Collapse.CollapseAnalysis(CFDir/'IDA_results.csv')
    collapse_result = ca.fit_collapse_fragility(fig_path=CFDir/'collapse_fragility.jpg')
    print(f"倒塌易损性中值 Sa: {collapse_result['median']:.3f} g")
    print(f"倒塌易损性对数标准差: {collapse_result['logstd']:.3f}")

    IDA_result_filtered = ca.filter_collapse()
    IDA_result_filtered.to_csv(CFDir/'IDA_results_filtered.csv', index=False, encoding='utf-8-sig')

    IDA.IDAAnalysis.plot_IDA_results(IDA_result, Stat=True, FigName=CFDir/'IDA.jpg')

    # IDA.plot_IDA_results_from_csv(CFDir/'IDA_results.csv', Stat=True, FigName=CFDir/'IDA.jpg')
