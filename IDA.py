########################################################
# Perform parallel IDA using Openseespy. Each record uses a single process.
# 
# Dependancy: 
# - openseespy, pandas, numpy, eqsig
########################################################

import copy
import multiprocessing as mp
import pandas as pd
import numpy as np
import eqsig.single
from pathlib import Path
import matplotlib.pyplot as plt

import MDOFOpenSees as mops
import ReadRecord

def plot_IDA_results(IDA_result: pd.DataFrame, EQRecordFile_list: list):
    fig, ax = plt.subplots()  
    for EQRecordFile in EQRecordFile_list:
        ind = (IDA_result['EQRecord']==EQRecordFile)
        ax.plot([max(drlist) for drlist in IDA_result['MaxDrift'][ind].values], 
            IDA_result['IM'][ind].values); 
    plt.show()

def IDA_1record(FEModel:mops.MDOFOpenSees, IM_list:list, EQRecordfile:str, period:float, 
    DeltaT = 'AsInRecord'):
    # Parameters:
    #   EQRecordfile:  earthquake record file. no file extension .unit: g
    #   IM_list: unit: g
    #   period: fundamental period
    #   DeltaT: time step of dynamic analyses

    FEModel.UniqueRecorderPrefix = 'URP'+ Path(EQRecordfile).name +'_'

    IDA_result = pd.DataFrame({'IM':[],'EQRecord':[],
        'MaxDrift':[],'MaxAbsAccel':[],'MaxRelativeAccel':[],'ResDrift':[],'Iffinish':[]})

    # calculate spectral acceleration
    p = Path(EQRecordfile)
    dt, nPts = ReadRecord.ReadRecord(EQRecordfile, (Path(p.parent,'temp_'+ p.name +'.dat')).as_posix())
    with open(Path(p.parent,'temp_'+ p.name +'.dat'), "r") as f:
        Allstr = f.read()
    Allstr = Allstr.split()
    Accel = np.array(Allstr).astype(float)
    record = eqsig.AccSignal(Accel * 9.8, dt)
    record.generate_response_spectrum(response_times=np.array([period]))
    SA = record.s_a[0]/9.8
        
    for IM in IM_list:
        Iffinish, tCurrent, TotalTime = FEModel.DynamicAnalysis(EQRecordfile, IM/SA, False, DeltaT)
        data = {'IM':IM,'EQRecord':EQRecordfile,'MaxDrift':[FEModel.MaxDrift.values],
            'MaxAbsAccel':[FEModel.MaxAbsAccel.values],'MaxRelativeAccel':[FEModel.MaxRelativeAccel.values],
            'ResDrift':FEModel.ResDrift,'Iffinish':Iffinish}
        IDA_result=pd.concat([IDA_result,pd.DataFrame(data)], ignore_index=True)

    return IDA_result

def IDA(FEModel:mops.MDOFOpenSees, IM_list:list, EQRecordfile_list:list, period:float,
    DeltaT = 'AsInRecord',NumPool = 1):

    IDA_result = pd.DataFrame({'IM':[],'EQRecord':[],
        'MaxDrift':[],'MaxAbsAccel':[],'MaxRelativeAccel':[],'ResDrift':[],'Iffinish':[]})

    if NumPool == 1:
        for EQRecordfile in EQRecordfile_list:
            FEModel_ = copy.deepcopy(FEModel)
            IDA_1RecordResult = IDA_1record(FEModel_,IM_list,EQRecordfile,period,DeltaT)
            IDA_result=pd.concat([IDA_result,IDA_1RecordResult], ignore_index=True)
    else:
        with mp.Pool(NumPool) as pool:
            IDA_1RecordResult_pool = [pool.apply_async(IDA_1record, 
                args=(copy.deepcopy(FEModel),IM_list,EQRecordfile,period,DeltaT,)) 
                for EQRecordfile in EQRecordfile_list]
            for IDA_1RecordResult in IDA_1RecordResult_pool:
                IDA_result = pd.concat([IDA_result,IDA_1RecordResult.get()], ignore_index=True) 

    return IDA_result