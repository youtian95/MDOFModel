########################################################
# Perform parallel IDA using Openseespy. Each record uses a single process.
# 
# Dependancy: 
# - openseespy, pandas, numpy, eqsig
########################################################

from collections import Counter
import copy
import multiprocessing as mp
import pandas as pd
import numpy as np
import eqsig.single
from pathlib import Path
import matplotlib.pyplot as plt

import MDOFOpenSees as mops
import ReadRecord

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
        data = {'IM':IM,'EQRecord':EQRecordfile,'MaxDrift':[FEModel.MaxDrift],
            'MaxAbsAccel':[FEModel.MaxAbsAccel],'MaxRelativeAccel':[FEModel.MaxRelativeAccel],
            'ResDrift':FEModel.ResDrift,'Iffinish':Iffinish}
        IDA_result=pd.concat([IDA_result,pd.DataFrame(data)], ignore_index=True)

    return IDA_result

def IDA_f(FEModel:mops.MDOFOpenSees, IM_list:list, EQRecordfile_list:list, period:float,
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

def SimulateEDPGivenIM(IDA_result:pd.DataFrame, IM_list:list, N_Sim, betaM:float = 0) -> pd.DataFrame:

    SimEDP = pd.DataFrame({'IM':[],'MaxDrift':[],'MaxAbsAccel':[],'ResDrift':[]})

    if isinstance(N_Sim,int):
        N_Sim = [N_Sim]*len(IM_list)

    # delete those IM whose EQ number is smaller than 3
    c = Counter(IDA_result['IM'].values.tolist())
    for keys in list(c.keys()):
        if c[keys] < 3:
            del(c[keys]) 
    IM_list_original = list(c.keys())
    # IM_list_original = [0]+IM_list_original

    if len(IM_list_original) == 0:
        return SimEDP

    # max EDP
    IDA_result = IDA_result[['IM','MaxDrift','MaxAbsAccel','ResDrift']]
    for i in range(0,  IDA_result.shape[0]):
        for j in range(0, IDA_result.shape[1]):
            IDA_result.iat[i,j] = np.array(IDA_result.iloc[i,j]).max()
    # newdf = pd.DataFrame([[0]*4], columns=list(IDA_result.columns))
    # IDA_result = pd.concat([newdf,IDA_result], ignore_index=True)

    # origional lnMean and lnbeta of EDP
    lnEDPs_mean_list_original = []
    lnEDPs_cov_list_original = []
    for IM in IM_list_original:
        EDPs = IDA_result.drop(columns=['IM'])[IDA_result['IM']==IM].values
        _,lnEDPs_mean,lnEDPs_cov,_,_,_ = IDA.FEMACodeSimulatingEDP(EDPs, betaM, 10)
        lnEDPs_mean_list_original.append(lnEDPs_mean)
        lnEDPs_cov_list_original.append(lnEDPs_cov)

    # simulate EDP
    assert len(IM_list)==len(N_Sim)
    for IM,N in zip(IM_list,N_Sim):
        lnEDPs_mean = IDA.interpMatrix(IM,IM_list_original,lnEDPs_mean_list_original)
        lnEDPs_cov = IDA.interpMatrix(IM,IM_list_original,lnEDPs_cov_list_original,True)
        if N<10:
            N_real = 10
        else:
            N_real = N
        W,_,_,_ = IDA.FEMACodeSimulatingEDPGivenlnMeanlncov(
            lnEDPs_mean,lnEDPs_cov,betaM,N_real)
        W = W[0:N,:]
        newdf = pd.DataFrame(np.concatenate((np.array([[IM]]*N),W),axis=1), 
            columns=list(SimEDP.columns))
        SimEDP = pd.concat([SimEDP,newdf], ignore_index=True)

    return SimEDP


class IDA():

    FEModel:mops.MDOFOpenSees = None
    IDA_result:pd.DataFrame = None

    def __init__(self, FEModel:mops.MDOFOpenSees):
        self.FEModel = FEModel

    def Analyze(self,IM_list:list, EQRecordfile_list:list, period:float,
        DeltaT = 'AsInRecord',NumPool = 1) -> pd.DataFrame:
        self.IDA_result = IDA_f(self.FEModel, IM_list, EQRecordfile_list, 
            period, DeltaT, NumPool)
        return self.IDA_result

    def plot_IDA_results(IDA_result:pd.DataFrame, Stat:bool = False):
        cm = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots()   # figsize=(8*cm, 6*cm)
        if not Stat:
            EQRecordFile_list = list(Counter(IDA_result['EQRecord'].values).keys())
            for EQRecordFile in EQRecordFile_list:
                ind = (IDA_result['EQRecord']==EQRecordFile)
                ax.plot([max(drlist) for drlist in IDA_result['MaxDrift'][ind].values], 
                    IDA_result['IM'][ind].values)
        else:
            IM_list = list(Counter(list(IDA_result['IM'].values)).keys())
            EDPmax_median = []
            EDPmax_1sigma_minus = []
            EDPmax_1sigma_plus = []
            for im in IM_list:
                EDP_values = [np.array(drlist).max() for drlist in 
                    IDA_result['MaxDrift'][IDA_result['IM']==im].values]
                EDPmax_median.append(np.exp(np.mean(np.log(EDP_values))))
                EDPmax_1sigma_minus.append(np.exp(np.log(EDPmax_median[-1]) - np.std(np.log(EDP_values))))
                EDPmax_1sigma_plus.append(np.exp(np.log(EDPmax_median[-1]) + np.std(np.log(EDP_values))))
            ax.plot(EDPmax_median,IM_list,'k',label='Median')
            ax.plot(EDPmax_1sigma_minus,IM_list,'b',label='-sigma')
            ax.plot(EDPmax_1sigma_plus,IM_list,'g',label='+sigma')

        

        plt.xlim(0,0.5)
        plt.ylim(0,2)

        plt.xticks(fontproperties = 'Times New Roman', fontsize=12)
        plt.yticks(np.arange(0, 2, 0.2), fontproperties = 'Times New Roman', fontsize=12)
        # 指定横纵坐标的字体以及字体大小，记住是fontsize不是size。yticks上我还用numpy指定了坐标轴的变化范围。

        plt.legend(loc='lower right', prop={'family':'Times New Roman', 'size':12})
        # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混

        # plt.title('1000 samples', fontdict={'family' : 'Times New Roman', 'size':12})
        # 指定图上标题的字体及大小

        plt.xlabel('Drift ratio', fontdict={'family' : 'Times New Roman', 'size':12})
        plt.ylabel('Spectral accelerations (g)', fontdict={'family' : 'Times New Roman', 'size':12})
        # 指定横纵坐标描述的字体及大小

        plt.savefig('IDA.eps', dpi=600, format='eps', bbox_inches="tight")
        # 保存文件，dpi指定保存文件的分辨率
        # bbox_inches="tight" 可以保存图上所有的信息，不会出现横纵坐标轴的描述存掉了的情况

        plt.show()

    def SimulateEDPGivenIM(self, IM_list:list, N_Sim, betaM:float = 0) -> pd.DataFrame:

        SimEDP = SimulateEDPGivenIM(self.IDA_result,IM_list,N_Sim,betaM)

        return SimEDP

    def interpMatrix(x,xp:list,Yp:list, nonnegative:bool = False)->np.array:
        # x: scalar
        # xp: list[float]
        # Yp: list[np.array]

        if len(xp)==1:
            xp = [0]+xp
            Yp = [0]+Yp

        inx = np.argsort(np.abs(x-np.array(xp)))

        Y = (Yp[inx[1]]-Yp[inx[0]])*(x-xp[inx[0]])/(xp[inx[1]]-xp[inx[0]]) + Yp[inx[0]]

        if nonnegative and (np.sum(Y<=0)>0):
            Y = Yp[inx[0]]
            if np.sum(Y<=0)>0:
                Y = Yp[inx[1]]
                
        return Y

    def FEMACodeSimulatingEDPGivenlnMeanlncov(lnEDPs_mean,lnEDPs_cov,betaM,num_realization):

        num_var = lnEDPs_cov.shape[1]

        # finding the rank of covariance matrix of lnEDPs. Calling it 
        # lnEDPs_cov_rank
        lnEDPs_cov_rank=np.linalg.matrix_rank(lnEDPs_cov)
        # inflating the variances with epistemic variability
        sigma = np.sqrt(np.diag(lnEDPs_cov))[:,np.newaxis] # sqrt first to avoid under/overflow
        sigmap2 = sigma * sigma
        R = lnEDPs_cov / (sigma @ (sigma.transpose())) 
        sigmap2 = sigmap2 + betaM**2    # Inflating variance for β m
        sigma=np.sqrt(sigmap2)
        sigma2 = sigma @ (sigma.T)
        lnEDPs_cov_inflated=R*sigma2

        # finding the eigenvalues eigenvectors of the covariance matrix. 
        # calling them D2_total and L_total
        D2_total,L_total = np.linalg.eig(lnEDPs_cov_inflated)
        idx = D2_total.argsort()
        D2_total = D2_total[idx]
        L_total = L_total[:,idx]
        
        # Partition L_total to L_use. L_use is the part of eigenvector matrix
        # L_total that corresponds to positive eigenvalues
        if lnEDPs_cov_rank >= num_var:
            L_use =L_total
        else:
            L_use = L_total[:, (num_var- lnEDPs_cov_rank):]
            # 因为L_use为特征值从小到大排列，所以0特征值在前面
            
        # Partition the D2_total to D2_use. D2_use is the part of eigenvalue
        # vector D2_total that corresponds to positive eigenvalues
        if lnEDPs_cov_rank >= num_var:
            D2_use = D2_total
        else:
            D2_use = D2_total[num_var- lnEDPs_cov_rank:]
        
        # Find the square root of D2_use and call is D_use. 
        # 创建对角矩阵
        D_use = np.diag(np.power(D2_use,0.5))

        # Generate Standard random numbers
        if lnEDPs_cov_rank >= num_var:
            U = np.random.normal(size=(num_realization, num_var))
        else:
            U = np.random.normal(size=(num_realization, lnEDPs_cov_rank))
            
        U = U.T

        # Create Lambda = D_use . L_use
        Lambda = L_use @ D_use
        # Create realizations matrix 
        Z = Lambda @ U + lnEDPs_mean @ np.ones((1,num_realization))
        lnEDPs_sim_mean = np.mean(Z,1)  # 行向量
        lnEDPs_sim_cov = np.cov(Z)
        ratio_mean = lnEDPs_sim_mean / (lnEDPs_mean.T)
        ratio_cov = lnEDPs_sim_cov / lnEDPs_cov
        W = np.exp(Z).T

        return W,R,ratio_mean,ratio_cov

    def FEMACodeSimulatingEDP(EDPs:np.array, betaM:float, num_realization):
        # Returns:
        #   W:  N_sim x N_var
        # 
        # Example usage:
        # W,lnEDPs_mean,R,ratio_mean,ratio_cov = FEMACodeSimulatingEDP(
        #     np.array([[1,2,4],[0.1,0.2,0.5],[8,9,10],[5,2,1]]),0.3,1000)

        EDPs = EDPs.astype(float)

        # taking natural logarithm of the EDPs. Calling it lnEDPs
        lnEDPs = np.log(EDPs)
        num_var = lnEDPs.shape[1]

        # finding the mean matrix of lnEDPs. Calling it lnEDPs_mean
        lnEDPs_mean = np.mean(lnEDPs,0)[:,np.newaxis]

        # finding the covariance matrix of lnEDPs. Calling it lnEDPs_cov
        lnEDPs_cov = np.cov(np.transpose(lnEDPs))

        W,R,ratio_mean,ratio_cov = IDA.FEMACodeSimulatingEDPGivenlnMeanlncov(
            lnEDPs_mean,lnEDPs_cov,betaM,num_realization)
        
        return W,lnEDPs_mean,lnEDPs_cov,R,ratio_mean,ratio_cov


