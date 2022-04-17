import IDA
import MDOF_LU as mlu
import MDOFOpenSees as mops
import pandas as pd
import numpy as np
from pathlib import Path
import time 

FEMAP695Dir = 'Resources/FEMA_P-695_far-field_ground_motions'
T:pd.DataFrame = pd.read_table(Path(FEMAP695Dir)/'MetaData.txt',sep=',')
EQRecordFile_list = [(Path(FEMAP695Dir)/str.replace(x,'.txt','')).as_posix()
    for x in T['AccelXfile'].to_list()] 

NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 3600, 'S2')
bld.set_DesignLevel('pre-code')
bld.OutputStructuralParameters('structural parameters')

fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)

if __name__ == '__main__':
    T1 = time.perf_counter()

    # IM_list = [0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0]
    IM_list = np.linspace(0.1,2.0,10).tolist()
    EQRecordFile_list = EQRecordFile_list
    IDA_obj = IDA.IDA(fe)
    IDA_result = IDA_obj.Analyze(IM_list, EQRecordFile_list, bld.T1,  NumPool=4) # DeltaT=0.1,

    # IM_list_sim = (np.linspace(0.1,2.0,10)+0.1).tolist()
    # IDA_result = IDA_obj.SimulateEDPGivenIM(IM_list_sim, bld.FloorArea, 0.25)

    T2 =time.perf_counter()

    print('Processing time %s sec' % ((T2 - T1)))

    IDA_result.to_csv('IDA_results.csv')

    IDA.IDA.plot_IDA_results(IDA_result, Stat=True)
