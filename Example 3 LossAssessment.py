import MDOF_LU as mlu
import MDOFOpenSees as mops
import BldLossAssessment as bl

NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 14709, 'S1L')
# bld.set_DesignLevel('pre-code')
# bld.OutputStructuralParameters('structural parameters')

fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.DynamicAnalysis('H-E12140', 3.0)
# fe.PlotForceDriftHistory(1)

blo = bl.BldLossAssessment(NumofStories,14709,'S1L','moderate-code','RES3')
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