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
import MDOF_CN as mcn
import MDOFOpenSees as mops
import IDA

# DesignInfo['Code'] = 'Hazus' / 'CN'
def main_IDA(IM_list,NumofStories,FloorArea,StructuralType,
    EQMetaDataFile, OutputCSVFile, SelfCenteringEnhancingFactor = 0,
    DesignInfo = {'Code': 'CN', 'SeismicDesignLevel': 'UNKNOWN', 'EQgroup': 'UNKNOWN', 'SiteClass': 'UNKNOWN'}):

    EQpath = Path(EQMetaDataFile)
    T:pd.DataFrame = pd.read_table(EQpath,sep=',')
    EQRecordFile_list = [(EQpath.parent/str.replace(x,'.txt','')).as_posix()
        for x in T['AccelXfile'].to_list()] 

    if DesignInfo['Code'] == 'Hazus':
        bld = mlu.MDOF_LU(NumofStories, FloorArea, StructuralType, 
                          SeismicDesignLevel=DesignInfo['SeismicDesignLevel'])
    elif DesignInfo['Code'] == 'CN':
        bld = mcn.MDOF_CN(NumofStories, FloorArea, StructuralType, 
            SeismicDesignLevel=DesignInfo['SeismicDesignLevel'], 
            EQGroup=DesignInfo['EQgroup'], 
            SiteClass=DesignInfo['SiteClass'])
    
    
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
    parser.add_argument('--DesignInfo', type=dict)
    parser.add_argument('--EQMetaDataFile')
    parser.add_argument('--OutputCSVFile',default = 'IDA_result.csv')
    parser.add_argument('--SelfCenteringEnhancingFactor',
        default = 0, type=float)
    args = parser.parse_args(args)

    if args.IM_list is None:
        print("ERROR: wrong arguments!")
        return

    main_IDA(args.IM_list,args.NumofStories,args.FloorArea,args.StructuralType,
        args.EQMetaDataFile,args.OutputCSVFile,args.SelfCenteringEnhancingFactor,
        args.DesignInfo)


# test function
# IM_list = [0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0]
# NumofStories = 2
# FloorArea = 5093.5
# StructuralType = 'C1'
# DesignInfo = {'Code': 'Hazus', 'SeismicDesignLevel': 'S1'}
# EQMetaDataFile = 'E:\CityResilienceAndResilientStructure\EQData\FEMA_P-695_far-field_ground_motions\MetaData_part10.txt'
# OutputCSVFile = 'E:\CityResilienceAndResilientStructure\IDA_results\IDA_results_SC05\IDA_result_ReprBldID_305.csv'
# SelfCenteringEnhancingFactor = 0.5
# main_IDA(IM_list,NumofStories,FloorArea,StructuralType,
#     EQMetaDataFile,OutputCSVFile,SelfCenteringEnhancingFactor,DesignInfo)

if __name__ == "__main__":
    main(sys.argv[1:])