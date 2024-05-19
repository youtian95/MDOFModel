import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parents[1]))

import MDOF_LU as mlu
import MDOF_CN as mcn
import MDOFOpenSees as mops

NumofStories = 3
bld = mcn.MDOF_CN(NumofStories, 1000, 'S2', City='石家庄',longitude=114.52,latitude=38.05)
# bld = mlu.MDOF_LU(NumofStories, 1000, 'S2M')
# bld.set_DesignLevel('pre-code')
bld.OutputStructuralParameters(os.path.join(os.path.dirname(__file__), 'structural parameters'))

fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.outputdir = os.path.dirname(__file__)
fe.DynamicAnalysis(os.path.join(os.path.dirname(__file__), 'H-E12140') , 3.0, True)

fe.PlotForceDriftHistory(1)