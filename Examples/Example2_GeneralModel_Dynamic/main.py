from pathlib import Path
import sys

examples_dir = str(Path(__file__).resolve().parent.parent)
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

from MDOFModel.models.GeneralModelWrapper import GeneralModelWrapper
from Example_MRF_Model import build_model

floor_nodes = [103, 203, 303, 403, 503, 603]             
story_heights = [5000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0]
base_nodes = [1, 2, 3, 4, 5]

CFDir = Path(__file__).resolve().parent / "Output"
CFDir.mkdir(parents=True, exist_ok=True)

wrapper_model = GeneralModelWrapper(
    build_model_func=build_model,
    floor_nodes=floor_nodes,
    story_heights=story_heights,
    dof=1,                 
    base_nodes=base_nodes,
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
