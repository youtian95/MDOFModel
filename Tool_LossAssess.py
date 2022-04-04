########################################################
# Estimate seismic losses based on time history dynamic analyses. 
# Python tools for cmd. 
# 
# Usage:
# 
# 
# Dependancy: 
# - pandas, numpy, openseespy
########################################################

import argparse
import sys
import os
from pathlib import Path

import pandas as pd

import MDOF_LU as mlu
import MDOFOpenSees as mops
import BldLossAssessment as bl

def DynamicAnalysis_1Sim(NumofStories,FloorArea,StructuralType,OccupancyClass,
    DesignLevel,EQRecordFile,EQScaling,OutputDir,SelfCenteringEnhancingFactor):

    bld = mlu.MDOF_LU(NumofStories, FloorArea, StructuralType)
    bld.set_DesignLevel(DesignLevel)
    # bld.OutputStructuralParameters('structural parameters')

    fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
        bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
    fe.SelfCenteringEnhancingFactor = SelfCenteringEnhancingFactor
    fe.DynamicAnalysis(EQRecordFile, EQScaling)
    # fe.PlotForceDriftHistory(1)

    blo = bl.BldLossAssessment(NumofStories, FloorArea,StructuralType,DesignLevel,OccupancyClass)
    blo.LossAssessment([fe.MaxDrift.max()],[fe.MaxAbsAccel.max()/9.8])  

    data = {
        'DS_Struct': blo.DS_Struct, 
        'DS_NonStruct_DriftSen':blo.DS_NonStruct_DriftSen,
        'DS_NonStruct_AccelSen': blo.DS_NonStruct_AccelSen,
        'RepairCost_Total': blo.RepairCost_Total,
        'RepairCost_Struct': blo.RepairCost_Struct,
        'RepairCost_NonStruct_DriftSen': blo.RepairCost_NonStruct_DriftSen,
        'RepairCost_NonStruct_AccelSen': blo.RepairCost_NonStruct_AccelSen,
        'RepairTime': blo.RepairTime,
        'RecoveryTime': blo.RecoveryTime,
        'FunctionLossTime': blo.FunctionLossTime
    }
    df = pd.DataFrame(data)
    df.to_csv(Path(OutputDir).joinpath('BldLoss.csv'),index=0)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumofStories',type=int)
    parser.add_argument('--FloorArea',type=float)
    parser.add_argument('--StructuralType')
    parser.add_argument('--OccupancyClass')
    parser.add_argument('--DesignLevel',default = 'moderate-code')
    parser.add_argument('--EQRecordFile')
    parser.add_argument('--EQScaling',default = 1.0,type=float)
    parser.add_argument('--OutputDir',default = '')
    parser.add_argument('--SelfCenteringEnhancingFactor',
        default = 0, type=float)
    args = parser.parse_args(args)

    DynamicAnalysis_1Sim(args.NumofStories,args.FloorArea,args.StructuralType,
        args.OccupancyClass,args.DesignLevel,args.EQRecordFile,
        args.EQScaling,args.OutputDir,args.SelfCenteringEnhancingFactor)

if __name__ == "__main__":
    main(sys.argv[1:])