from pathlib import Path
import os
import sys

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MDOFModel import MDOF_LU as mlu
from MDOFModel import MDOFOpenSees as mops

NumOfStories = 9
bld = mlu.MDOF_LU(NumOfStories, 45.75*45.75, 'S1H')
bld.set_DesignLevel('high-code')
bld.OutputStructuralParameters(os.path.join(os.path.dirname(__file__), 'structural parameters'))

fe = mops.MDOFOpenSees(NumOfStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.SelfCenteringEnhancingFactor = 0.0
fe.outputdir = os.path.dirname(__file__)
# dispacement control
# D_ult = bld.DeltaCi[0]
# maxU = [0.1*D_ult,-0.1*D_ult,0.2*D_ult,-0.2*D_ult,D_ult,-D_ult,2*D_ult,-2*D_ult,0]
maxU = [9*3.6*0.08]
fe.StaticPushover(maxU,dU = 0.01, CFloor = 9)

fe.PlotForceDriftHistory(1)