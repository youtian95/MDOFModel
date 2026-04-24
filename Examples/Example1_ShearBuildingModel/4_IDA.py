from pathlib import Path
import time 
import sys
import pandas as pd
import numpy as np

from MDOFModel.analysis import IDA
from MDOFModel.models import MDOF_LU as mlu
from MDOFModel.models import MDOFOpenSees as mops
import MDOFModel

CFDir = Path(__file__).resolve().parent / "Example5_Output"
CFDir.mkdir(parents=True, exist_ok=True)



NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 3600, 'S2')
bld.set_DesignLevel('pre-code')
bld.OutputStructuralParameters(str(CFDir/'structural parameters'))

fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.outputdir = str(CFDir)

if __name__ == '__main__':
    T1 = time.perf_counter()

    # IM_list = [0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0]
    IM_list = np.linspace(0.1,2.0,10).tolist()
    IDA_obj = IDA.IDAAnalysis(fe)
    IDA_result = IDA_obj.Analyze(IM_list, period=bld.T1, NumPool=4) # DeltaT=0.1,

    # IM_list_sim = (np.linspace(0.1,2.0,10)+0.1).tolist()
    # IDA_result = IDA_obj.SimulateEDPGivenIM(IM_list_sim, bld.FloorArea, 0.25)

    T2 =time.perf_counter()

    print('Processing time %s sec' % ((T2 - T1)))

    IDA_result.to_csv(str(CFDir/'IDA_results.csv'))

    IDA.IDAAnalysis.plot_IDA_results(IDA_result, Stat=True, FigName=str(CFDir/'IDA.jpg'))
