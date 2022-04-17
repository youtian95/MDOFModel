import MDOF_LU as mlu
import MDOFOpenSees as mops

NumOfStories = 3
bld = mlu.MDOF_LU(NumOfStories, 3600, 'C1')
bld.set_DesignLevel('pre-code')
bld.OutputStructuralParameters('structural parameters')

fe = mops.MDOFOpenSees(NumOfStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.SelfCenteringEnhancingFactor = 0.75
D_ult = bld.DeltaCi[0]
maxU = [0.1*D_ult,-0.1*D_ult,0.2*D_ult,-0.2*D_ult,D_ult,-D_ult,2*D_ult,-2*D_ult,0]
fe.StaticPushover(maxU,dU = 0.000002, CFloor = 1)

fe.PlotForceDriftHistory(1)