from pathlib import Path
import time 
import sys
import pandas as pd
import numpy as np

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MDOFModel import IDA
from MDOFModel import MDOF_LU as mlu
from MDOFModel import MDOFOpenSees as mops
import MDOFModel

CFDir = Path(__file__).resolve().parent

FEMAP695Dir = str(Path(MDOFModel.__file__).parent / 'Resources/FEMA_P-695_far-field_ground_motions')
T:pd.DataFrame = pd.read_table(str(Path(FEMAP695Dir)/'MetaData.txt'),sep=',')
EQRecordFile_list = [str(Path(FEMAP695Dir)/str.replace(x,'.txt',''))
    for x in T['AccelXfile'].to_list()] 

NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 3600, 'S2')
bld.set_DesignLevel('pre-code')
bld.OutputStructuralParameters(str(CFDir/'structural parameters'))

fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.outputdir = str(Path(__file__).resolve().parent)

if __name__ == '__main__':
    T1 = time.perf_counter()

    # IM_list = [0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0]
    IM_list = np.linspace(0.1,2.0,10).tolist()
    IDA_obj = IDA.IDA(fe)
    IDA_result = IDA_obj.Analyze(IM_list, EQRecordFile_list, bld.T1,  NumPool=4) # DeltaT=0.1,

    # IM_list_sim = (np.linspace(0.1,2.0,10)+0.1).tolist()
    # IDA_result = IDA_obj.SimulateEDPGivenIM(IM_list_sim, bld.FloorArea, 0.25)

    T2 =time.perf_counter()

    print('Processing time %s sec' % ((T2 - T1)))

    IDA_result.to_csv(str(CFDir/'IDA_results.csv'))

    IDA.IDA.plot_IDA_results(IDA_result, Stat=True, FigName=str(CFDir/'IDA.jpg'))
