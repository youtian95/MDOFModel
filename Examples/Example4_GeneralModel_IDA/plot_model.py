"""
Example: 使用 GeneralModelWrapper 绘制模型示意图、重力变形图和模态振型图。
运行方式：直接执行本脚本即可，无需多进程保护。
"""
from pathlib import Path
import sys

parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

examples_dir = str(Path(__file__).resolve().parent.parent)
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

from MDOFModel.models.GeneralModelWrapper import GeneralModelWrapper
from Example_MRF_Model import build_model

floor_nodes = [103, 203, 303, 403, 503, 603]
story_heights = [5000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0]
base_nodes = [1, 2, 3, 4, 5]

wrapper_model = GeneralModelWrapper(
    build_model_func=build_model,
    floor_nodes=floor_nodes,
    story_heights=story_heights,
    dof=1,
    base_nodes=base_nodes,
    g_factor=9800.0,
)

# 绘制模型节点与单元示意图
wrapper_model.PlotModel()

# 绘制施加重力后的变形图
wrapper_model.Plot_Gravity_def()

# 绘制第一阶模态振型动画
wrapper_model.Plot_Mode_Shape(mode_no=1)
