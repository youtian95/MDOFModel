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
    
    print("====== Running GeneralModelWrapper Cyclic Pushover Analysis ======")
    max_disp = 1000.0
    # One full cycle: 0 → +max → -max → +max → 0
    wrapper_model.StaticPushover(
        maxU=[max_disp, -max_disp, max_disp, 0.0],
        dU=10.0,
        CFloor='roof',
        animate=True
    )
