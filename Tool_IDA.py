########################################################
# Perform IDA and create a result file for a building
# 
# Usage:
# 
# 
# Dependancy: 
# - pandas, numpy, openseespy
########################################################

import argparse
import sys
from pathlib import Path

import pandas as pd

import MDOF_LU as mlu
import MDOFOpenSees as mops
import IDA

def main_IDA(IM_list,NumofStories,FloorArea,StructuralType,OccupancyClass,
    DesignLevel,EQMetaDataFile,OutputCSVFile,SelfCenteringEnhancingFactor):

    EQpath = Path(EQMetaDataFile)
    T:pd.DataFrame = pd.read_table(EQpath,sep=',')
    EQRecordFile_list = [(EQpath.parent/str.replace(x,'.txt','')).as_posix()
        for x in T['AccelXfile'].to_list()] 

    bld = mlu.MDOF_LU(NumofStories, FloorArea, StructuralType)
    bld.set_DesignLevel(DesignLevel)
    
    fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
        bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
    fe.SelfCenteringEnhancingFactor = SelfCenteringEnhancingFactor

    IDA_obj = IDA.IDA(fe)
    IDA_result = IDA_obj.Analyze(IM_list, EQRecordFile_list, bld.T1, DeltaT=0.1, NumPool=4)

    IDA_result.to_csv(Path(OutputCSVFile))

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--IM_list',nargs='+',type=float)
    parser.add_argument('--NumofStories',type=int)
    parser.add_argument('--FloorArea',type=float)
    parser.add_argument('--StructuralType')
    parser.add_argument('--OccupancyClass')
    parser.add_argument('--DesignLevel',default = 'moderate-code')
    parser.add_argument('--EQMetaDataFile')
    parser.add_argument('--OutputCSVFile',default = 'IDA_result.csv')
    parser.add_argument('--SelfCenteringEnhancingFactor',
        default = 0, type=float)
    args = parser.parse_args(args)

    main_IDA(args.IM_list,args.NumofStories,args.FloorArea,args.StructuralType,
        args.OccupancyClass,args.DesignLevel,args.EQMetaDataFile,
        args.OutputCSVFile,args.SelfCenteringEnhancingFactor)

if __name__ == "__main__":
    main(sys.argv[1:])