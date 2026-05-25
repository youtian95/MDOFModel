########################################################
# 对建筑执行增量动力分析（IDA）并输出结果文件。
########################################################

import argparse
import sys
from pathlib import Path
import pandas as pd

from ..models import MDOF_LU as mlu
from ..models import MDOF_CN as mcn
from ..models import MDOFOpenSees as mops
from . import IDA_2D as IDA

# DesignInfo['Code'] = 'Hazus' / 'CN'
def main_IDA(IM_list,NumofStories,FloorArea,StructuralType,
    EQMetaDataFile, OutputCSVFile, SelfCenteringEnhancingFactor = 0,
    DesignInfo = {'Code': 'CN', 'SeismicDesignLevel': 'UNKNOWN', 'EQgroup': 'UNKNOWN', 'SiteClass': 'UNKNOWN'}, NumPool = 1, TempDir = Path.cwd()/'temp', UseRelativeIM = False, WriteStructParaFile = None):
    '''
    对建筑执行增量动力分析。
    参数:
        IM_list (list): 强度指标 Sa(T1) 列表，单位: g。
            若 UseRelativeIM 为 True，则为相对于 475 年重现期 Sa(T1) 的强度指标列表。
        NumofStories (int): 层数
        FloorArea (float): 建筑面积，单位: m^2
        StructuralType (str): 结构类型
        EQMetaDataFile (Path): 地震动元数据文件路径
        OutputCSVFile (Path): IDA 结果输出 CSV 文件路径
        SelfCenteringEnhancingFactor (float): 自复位增强系数
        DesignInfo (dict): 设计信息字典
        NumPool (int): 并行进程数
        TempDir (Path): OpenSees 分析临时文件目录
        UseRelativeIM (bool): 为 True 时， IM_list 为相对于 475 年重现期 Sa(T1) 的强度指标
        WriteStructParaFile (Path): 结构参数输出文件路径。为 None 时不输出。
    '''

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
    else:
        raise Exception('Design code not supported!')
    
    if WriteStructParaFile is not None:
        bld.OutputStructuralParameters(WriteStructParaFile)
    
    fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
        bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
    if not TempDir.exists():
        TempDir.mkdir(parents=True, exist_ok=True)
    fe.outputdir = TempDir
    fe.SelfCenteringEnhancingFactor = SelfCenteringEnhancingFactor

    # 若使用相对强度指标，将 IM_list 乘以 475 年重现期的 Sa(T1)
    if UseRelativeIM:
        if DesignInfo['Code'] == 'Hazus':
            Sa_T1 = bld.Cs
        elif DesignInfo['Code'] == 'CN':
            Sa_T1 = bld.Sa_T1
        IM_list = [IM_list[i]*Sa_T1 for i in range(len(IM_list))]

    IDA_obj = IDA.IDAAnalysis(fe)
    IDA_result = IDA_obj.Analyze(IM_list, EQRecordFile_list, bld.T1, DeltaT=0.1, NumPool=NumPool)

    IDA_result.to_csv(Path(OutputCSVFile), index=False, encoding='utf-8-sig')

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--IM_list',nargs='+',type=float)
    parser.add_argument('--NumofStories',type=int)
    parser.add_argument('--FloorArea',type=float)
    parser.add_argument('--StructuralType')
    # Replace the dictionary argument with individual parameters
    parser.add_argument('--DesignCode', default='CN', choices=['CN', 'Hazus'],
                        help='Design code: CN or Hazus')
    parser.add_argument('--DesignLevel', default='UNKNOWN')
    parser.add_argument('--EQgroup', default='UNKNOWN')
    parser.add_argument('--SiteClass', default='UNKNOWN')
    # other parameters
    parser.add_argument('--EQMetaDataFile')
    parser.add_argument('--OutputCSVFile',default = 'IDA_result.csv')
    parser.add_argument('--SelfCenteringEnhancingFactor',
        default = 0, type=float)
    parser.add_argument('--NumPool',default = 1, type=int)
    parser.add_argument('--UseRelativeIM',default = False, type=bool)
    args = parser.parse_args(args)

    # 将各个参数组装为 DesignInfo 字典
    DesignInfo = {
        'Code': args.DesignCode,
        'SeismicDesignLevel': args.DesignLevel,
        'EQgroup': args.EQgroup,
        'SiteClass': args.SiteClass
    }

    if args.IM_list is None:
        print("ERROR: wrong arguments!")
        return

    main_IDA(args.IM_list,args.NumofStories,args.FloorArea,args.StructuralType,
        args.EQMetaDataFile,args.OutputCSVFile,args.SelfCenteringEnhancingFactor,
        DesignInfo,
        NumPool = args.NumPool, UseRelativeIM = args.UseRelativeIM)


# 测试函数
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