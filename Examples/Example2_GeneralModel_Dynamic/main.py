from pathlib import Path
import sys

examples_dir = str(Path(__file__).resolve().parent.parent)
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

from MDOFModel.models.GeneralModelWrapper import GeneralModelWrapper
from Example_6Story_MRF_Model import BASE_NODES, FLOOR_NODES, STORY_HEIGHTS, build_model

CFDir = Path(__file__).resolve().parent / "Output"
CFDir.mkdir(parents=True, exist_ok=True)

wrapper_model = GeneralModelWrapper(
    build_model_func=build_model,
    floor_nodes=FLOOR_NODES,
    story_heights=STORY_HEIGHTS,
    dof=1,                 
    base_nodes=BASE_NODES,
    g_factor=9800.0,
)

if __name__ == '__main__':
    record_file = str(Path(__file__).resolve().parent / 'H-E12140')
    
    print("====== Running GeneralModelWrapper Dynamic Analysis ======")
    wrapper_model.DynamicAnalysis(
        record_file=record_file,
        scale_factor=3.0,
        animate=True,
        show_progress=True
    )
    
    print("Max Drift:", wrapper_model.MaxDrift)
    print("Max Abs Accel:", wrapper_model.MaxAbsAccel)
    print("Residual Drift:", wrapper_model.ResDrift)
