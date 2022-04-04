import IDA
import MDOF_LU as mlu
import MDOFOpenSees as mops
import pandas as pd
import numpy as np
from pathlib import Path
import time 
import matplotlib.pyplot as plt

FEMAP695Dir = 'Resources/FEMA_P-695_far-field_ground_motions'
T:pd.DataFrame = pd.read_table(Path(FEMAP695Dir)/'MetaData.txt',sep=',')
EQRecordFile_list = [(Path(FEMAP695Dir)/str.replace(x,'.txt','')).as_posix()
    for x in T['AccelXfile'].to_list()] 

NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 1000, 'S2M')
# bld.set_DesignLevel('pre-code')
bld.OutputStructuralParameters('structural parameters')

fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)

if __name__ == '__main__':
    T1 = time.perf_counter()
    IM_list = np.linspace(0.1,2.0,10).tolist()
    EQRecordFile_list = EQRecordFile_list
    IDA_result = IDA.IDA(fe, IM_list, EQRecordFile_list,
        bld.T1, DeltaT=0.1, NumPool=4)
    T2 =time.perf_counter()
    print('Processing time %s sec' % ((T2 - T1)))
    IDA_result.to_csv('IDA_results.csv')
    IDA.plot_IDA_results(IDA_result, EQRecordFile_list)
