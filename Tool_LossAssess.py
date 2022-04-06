########################################################
# Estimate seismic losses.
# Python tools for cmd. 
# 
# Usage:
# 1. loss assessment based on a dynamic analysis
#   --EQRecordFile <> --EQScaling <> --NumofStories <> --FloorArea <> --StructuralType <>
#       --OccupancyClass <> --DesignLevel <> --OutputDir <> --SelfCenteringEnhancingFactor <>
# 
# 2. Simulate EDP given a IM based on IDA results.
#   --IM_list <0.1 0.2 0.3 ...> --N_Sim <100> --IDA_result <> --betaM <>
#       --OutputDir <> --NumofStories <> --FloorArea <> --StructuralType <>
#       --OccupancyClass <> --DesignLevel <>
# 
# Dependancy: 
# - pandas, numpy, openseespy
########################################################

import argparse
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

import MDOF_LU as mlu
import MDOFOpenSees as mops
import BldLossAssessment as bl
import IDA

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

def Simulate_losses_given_IM_basedon_IDA(IDA_result,IM_list,N_Sim,betaM,OutputDir,
    NumofStories,FloorArea,StructuralType,DesignLevel,OccupancyClass):

    IDA_result = pd.read_csv(Path(IDA_result))
    IDA_result = IDA_result.loc[:, ~IDA_result.columns.str.contains('^Unnamed')]
    for ind,row in IDA_result.iterrows():
        for varname in ['MaxDrift','MaxAbsAccel','MaxRelativeAccel']:
            if isinstance(row[varname],str):
                IDA_result.at[ind,varname] = np.array(
                    [float(val) for val in row[varname].replace('[','').replace(']','').split()])

    if len(N_Sim)==1:
        N_Sim = N_Sim[0]
    SimEDP = IDA.SimulateEDPGivenIM(IDA_result,IM_list,N_Sim,betaM)
    SimEDP.to_csv(Path(OutputDir)/'SimEDP.csv')

    blo = bl.BldLossAssessment(NumofStories, FloorArea,StructuralType,DesignLevel,OccupancyClass)
    blo.LossAssessment(SimEDP['MaxDrift'].tolist(),(SimEDP['MaxAbsAccel']/9.8).tolist())  
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
    df.to_csv(Path(OutputDir)/'BldLoss.csv')


def main(args):
    parser = argparse.ArgumentParser()
    
    # usage 1
    parser.add_argument('--EQRecordFile')
    parser.add_argument('--EQScaling',default = 1.0,type=float)
    parser.add_argument('--SelfCenteringEnhancingFactor',
        default = 0, type=float)

    # usage 2   
    parser.add_argument('--IM_list', nargs='+', type=float)
    parser.add_argument('--N_Sim', nargs='+', type=int)
    parser.add_argument('--IDA_result')
    parser.add_argument('--betaM', type=float, default=0.0)

    # common arguments
    parser.add_argument('--OutputDir',default = '')
    parser.add_argument('--NumofStories',type=int)
    parser.add_argument('--FloorArea',type=float)
    parser.add_argument('--StructuralType')
    parser.add_argument('--OccupancyClass')
    parser.add_argument('--DesignLevel',default = 'moderate-code')

    args = parser.parse_args(args)

    if not args.EQRecordFile is None:
        DynamicAnalysis_1Sim(args.NumofStories,args.FloorArea,args.StructuralType,
            args.OccupancyClass,args.DesignLevel,args.EQRecordFile,
            args.EQScaling,args.OutputDir,args.SelfCenteringEnhancingFactor)
    elif not args.IDA_result is None:
        Simulate_losses_given_IM_basedon_IDA(args.IDA_result,args.IM_list,args.N_Sim,
            args.betaM,args.OutputDir,
            args.NumofStories,args.FloorArea,args.StructuralType,args.DesignLevel,args.OccupancyClass)
    else:
        print('ERROR: wrong arguments')

if __name__ == "__main__":
    main(sys.argv[1:])