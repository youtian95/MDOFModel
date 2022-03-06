import MDOF_LU as mlu
import MDOFOpenSees as mops

NumOfStories = 3
bld = mlu.MDOF_LU(NumOfStories, 1000, 'S2M')
# bld.set_DesignLevel('pre-code')
bld.OutputStructuralParameters('structural parameters')

fe = mops.MDOFOpenSees(NumOfStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
D_ult = bld.DeltaCi[0]
# fe.StaticPushover(maxU = [0.2*D_ult], 
#     dU = 0.000001, CFloor = 1)
fe.StaticPushover(maxU = [0.2*D_ult,-0.2*D_ult,D_ult,-D_ult,2*D_ult,-2*D_ult,0], 
    dU = 0.00001, CFloor = 1)

fe.PlotForceDriftHistory(1)