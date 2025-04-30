from pathlib import Path
import os
import sys

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MDOFModel import MDOF_LU as mlu
from MDOFModel import MDOFOpenSees as mops
from MDOFModel import BldLossAssessment as bl

NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 3600, 'S2')
bld.set_DesignLevel('pre-code')

fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.outputdir = os.path.dirname(__file__)
fe.DynamicAnalysis(os.path.join(os.path.dirname(__file__), 'H-E12140'), 3.0)
# fe.PlotForceDriftHistory(1)

blo = bl.BldLossAssessment(NumofStories,bld.FloorArea,bld.StructuralType,bld.getDesignLevel(),'RES3')
blo.LossAssessment([fe.MaxDrift.max()],[fe.MaxAbsAccel.max()/9.8])  
print(blo.DS_Struct)
print(blo.DS_NonStruct_DriftSen)
print(blo.DS_NonStruct_AccelSen)
print(blo.RepairCost_Total)
print(blo.RepairCost_Struct)
print(blo.RepairCost_NonStruct_DriftSen)
print(blo.RepairCost_NonStruct_AccelSen)
print(blo.RepairTime)
print(blo.RecoveryTime)
print(blo.FunctionLossTime)