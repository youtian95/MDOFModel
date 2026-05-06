########################################################
# 使用 OpenSeesPy 执行动力分析（SI 单位制）
########################################################

from ctypes import Union
import matplotlib.pyplot as plt
from cmath import pi
from openseespy.opensees import *
import pandas as pd
import numpy as np
from pathlib import Path
import os
import mpl_toolkits.axisartist as axisartist

from ..analysis import ReadRecord

class MDOFOpenSees():

    UniqueRecorderPrefix = 'URP0_'
    __g = 9.8

    # 结构参数
    NStories : int = 0
    m: list = []
    k: list = []
    DampingRatio:float = 0.05
    HystereticCurveType: str = 'Elastic'
    HystereticParameters = ()
    SelfCenteringEnhancingFactor = 0.0 # 0-1

    # 输出目录
    outputdir = str(Path.cwd())

    # 执行推覆分析的结果保存
    # DriftHistory = {} # DriftHistory['time'] 为时间列表，DriftHistory[1] 为第1层层间位移角列表
    # ForceHistory = {} 
    NodeDispHistory = {} # NodeDispHistory['time'], NodeDispHistory[1-N]

    # 动力分析结果
    MaxDrift = np.array([]) # MaxDrift[0] 为第1层
    MaxAbsAccel = np.array([]) # MaxAbsAccel[0] 为地面
    MaxRelativeAccel = np.array([]) # [0] 为地面
    MaxAbsVel = np.array([]) # MaxAbsVel[0] 为地面（固定节点，值为 0）
    ResDrift = None
    DriftHistory = {} # DriftHistory['time'] 为时间列表，DriftHistory[1] 为第1层层间位移角列表
    ForceHistory = {} 
    NodeAbsAccelHistory = {} # NodeAbsAccelHistory[0] 为地面
    NodeRelativeAccelHistory = {}


    def __init__(self, NStories :int, m: list, k:list, DampingRatio:float,
        HystereticCurveType: str, *HystereticParameters):
        # -NStories: 层数
        # -m: 各层质量列表，单位 kg
        # -k: 各层弹性层刚度列表，单位 N/m
        # -DampingRatio: 阻尼比
        # -HystereticCurveType: ['Elastic','Modified-Clough','Kinematic hardening','Pinching']
        # -*HystereticParameters: 可变参数，包括 (Vyi, betai, etai, DeltaCi, tao)
        # {
        #   -Vyi, yield shear force for each story, N
        #   -betai, overstrength ratio of ultimate strength to yield strength for each story
        #   -etai, hardening ratio for each story
        #   -DeltaCi, displacement threshold for complete damage state, m
        #   -tao, degradation factor
        # }

        self.NStories = NStories
        self.m = m
        self.k = k
        self.DampingRatio = DampingRatio
        self.HystereticCurveType = HystereticCurveType
        self.HystereticParameters = HystereticParameters

    def StaticPushover(self, maxU: list = [0.10,-0.10,0], dU = 0.001,
        CFloor = 'roof', ifprint: bool = True):
        # 参数:
        # maxU   - 目标位移，单位 m
        # dU     - 位移增量，单位 m
        # CFloor - 控制层
        #
        # 返回值:
        # Iffinish, currentDisp
        
        if ifprint:
            print('Pushover analysis of a MDOF lumped-mass building model with OpenSees...')
        
        self.__BuildModel(ifprint)

        tsTag = 301
        timeSeries('Linear', tsTag)
        patternTag = 101
        pattern('Plain', patternTag, tsTag)

        # Create nodal loads
        #    nd    FX  FY  MZ
        for i in range(1,self.NStories+1):
            load(i, i, 0.0, 0.0)

        # recorders
        outputdir = Path(self.outputdir).relative_to(Path.cwd())
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')), '-time',
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), '-time',
            '-ele', *list(range(1,self.NStories+1)), 'axialForce')
        recorder('Node', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'NodeDispHistory.txt')),'-time',
            '-node', *list(range(1,self.NStories+1)), '-dof', 1, 'disp')
        
        # 执行分析        
        Tol = 1e-6
        maxNumIter = 100
        if isinstance(CFloor,str) & (CFloor == 'roof'):
            CFloor = self.NStories
        system('FullGeneral')
        constraints('Transformation')
        numberer('RCM')
        test('NormDispIncr', Tol, maxNumIter)
        algorithm('NewtonLineSearch') 

        Test = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 3:'EnergyIncr', 
            4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
        Algorithm = {1:'KrylovNewton', 2: 'SecantNewton' , 3:'ModifiedNewton' , 
            4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

        currentDisp = 0.0
        ok = 0

        for i in range(len(maxU)):
            while ok == 0 and abs(currentDisp-maxU[i])>dU:
                numIter=100
                integrator('DisplacementControl', CFloor, 1, 
                    np.sign(maxU[i]-currentDisp)*dU, numIter)
                analysis('Static')
                ok = analyze(1)
                # 分析失败时跳出
                if ok != 0:
                    break
                currentDisp = nodeDisp(CFloor, 1)

        Iffinish = not ok

        if ifprint:
            print(f'State (Successful or Fault): {Iffinish:d}')
        
        wipe()
        self.__ReadPushoverRecorderFiles()

        return Iffinish, currentDisp
        
    def DynamicAnalysis(self, EQRecordfile:str, GMScaling:float, ifprint: bool = True,
        DeltaT = 0.1):
        # 参数:
        # -ifprint:       是否打印过程信息
        # -EQRecordfile:  PEER 格式地震动记录文件，例如 'H-E12140'
        # -GMScaling:     地震动缩放系数
        # -DeltaT:        'AsInRecord' 或浮点数
        #
        # 返回值:
        # Iffinish, tCurrent, TotalTime

        if ifprint:
            print('Perform dynamic analysis of a MDOF lumped-mass building model with OpenSees...')

        self.__BuildModel(ifprint)

        # 将 SMD 记录转换为 OpenSees 可读格式
        p = Path(EQRecordfile)
        dt, nPts = ReadRecord.ReadRecord(EQRecordfile, 
            (Path(p.parent, self.UniqueRecorderPrefix + p.name +'.dat')).as_posix())

        # 均匀激励：加速度输入
        tsTag = 100
        EQfile = Path(p.parent,self.UniqueRecorderPrefix + p.name +'.dat')
        timeSeries('Path', tsTag, '-dt', dt, '-filePath', 
            os.path.relpath(EQfile,Path.cwd()),
            '-factor', self.__g * GMScaling) # 用相对路径，避免路径中有中文字符
        IDloadTag = 400			# load tag
        GMdirection = 1
        pattern('UniformExcitation', IDloadTag, GMdirection, '-accel', tsTag)

        # recorders
        outputdir = Path(self.outputdir).relative_to(Path.cwd())
        recorder('EnvelopeElement', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'MaxDrift.txt')),
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')),'-time',
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), '-time',
            '-ele', *list(range(1,self.NStories+1)), 'axialForce')
        recorder('EnvelopeNode', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'MaxAbsAccel.txt')), 
            '-timeSeries', tsTag, 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('EnvelopeNode', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'MaxRelativeAccel.txt')),
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('EnvelopeNode', '-file',
            str(Path(outputdir,self.UniqueRecorderPrefix+'MaxAbsVel.txt')),
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'vel')
        recorder('Node', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'NodeAbsAccelHistory.txt')),
            '-timeSeries', tsTag, '-time', 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('Node', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'NodeRelativeAccelHistory.txt')), '-time', 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')


        # 动力分析
        Tol = 1e-8
        maxNumIter = 10
        DtAnalysis = dt if DeltaT== 'AsInRecord' else DeltaT # dt
        wipeAnalysis()
        constraints('Transformation')
        numberer('RCM')
        # system('UmfPack') # 仅在使用 ExpressNewton 算法时有效
        system('BandGeneral')

        tCurrent = getTime()

        Test = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 3:'EnergyIncr', 
            4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
        Algorithm = {8: 'NewtonLineSearch', 1:'KrylovNewton', 2: 'SecantNewton' , 3:'ModifiedNewton' , 
            4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden'} # 9: 'ExpressNewton', 

        # algorithm ExpressNewton 2 1.0 -currentTangent -factorOnce

        tFinal = nPts*dt

        time = [tCurrent]
        ok = 0
        while tCurrent < tFinal:   
            for i in Test:
                test(Test[i], Tol, maxNumIter)    
                for j in Algorithm: 
                    if j==9:
                        algorithm(Algorithm[j], 2, 1.0, '-currentTangent','-factorOnce')
                    elif j < 4:
                        algorithm(Algorithm[j], '-initial')
                    else:
                        algorithm(Algorithm[j])
                    while ok == 0 and tCurrent < tFinal:    
                        NewmarkGamma = 0.5
                        NewmarkBeta = 0.25
                        integrator('Newmark', NewmarkGamma, NewmarkBeta)
                        analysis('Transient')
                        ok = analyze(1, DtAnalysis)
                        if ok == 0:
                            tCurrent = getTime()                
                            time.append(tCurrent)
            break

        Iffinish = not ok
        TotalTime = tFinal

        if ifprint:
            print(f'State (Successful or Fault): {Iffinish:d}')
            print(f'The analysis ends at {tCurrent:.3f} sec out of {TotalTime:.3f} sec.')
        
        wipe()
        self.__ReadDynamicRecorderFiles()

        return Iffinish, tCurrent, TotalTime

    def PlotForceDriftHistory(self, NumOfStory:int = 1):
        cm = 1/2.54  # centimeters in inches
        fig = plt.figure('Origional',(10*cm,8*cm))
    
        ax = axisartist.Subplot(fig, 1,1,1)
        fig.add_axes(ax)
        
        ax.axis[:].set_visible(False)
        
        ax.axis["x"] = ax.new_floating_axis(0, 0)
        ax.axis["y"] = ax.new_floating_axis(1, 0)
        ax.axis["x"].set_axis_direction('top')
        ax.axis["y"].set_axis_direction('left')
        ax.axis["x"].set_axisline_style("->", size = 2.0)
        ax.axis["y"].set_axisline_style("->", size = 2.0)
        
        ax.plot(self.DriftHistory[NumOfStory],self.ForceHistory[NumOfStory],linewidth = 2)
        # plt.title('y = 2sin(2t)',fontsize = 14, pad = 20)
        
        # ax.set_xticks(np.linspace(0.25,1.25,5)*np.pi)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # ax.set_xticklabels(['$\\frac{\pi}{4}$','$\\frac{\pi}{2}$', '$\\frac{3\pi}{4}$', '$\pi$', '$\\frac{5\pi}{4}$', '$\\frac{3\pi}{2}$'])
        # ax.set_yticks([0, 1, 2])
        
        # ax.set_xlim(-0.5*np.pi,1.5*np.pi)
        # ax.set_ylim(-2.2, 2.2)
        
        plt.show()


    def __BuildModel(self, ifprint: bool):
        # 建立建筑模型

        wipe()			
        model('basic', '-ndm', 2, '-ndf', 3)

        storyLength = 1.0

        # 定义节点
        node(0, 0., 0.)
        fix(0, 1, 1, 1) 
        for i in range(self.NStories):
            node(i+1, (i+1)*storyLength, 0.)
            mass(i+1, self.m[i], 0., 0.)
            fix(i+1, 0, 1, 1) 
        
        # 定义材料
        E = 1.0
        matTag = [i+1 for i in range(self.NStories)]
        A = [0] * self.NStories
        for i in range(self.NStories):
            A[i] = self.k[i] * storyLength / E
            # *HystereticParameters = (Vyi, betai, etai, DeltaCi, tao)
            if self.HystereticCurveType == 'Elastic':
                uniaxialMaterial(self.HystereticCurveType, matTag[i], E)
            elif self.HystereticCurveType in ['Modified-Clough','Kinematic hardening','Pinching']:
                Vyi = self.HystereticParameters[0][i]
                betai = self.HystereticParameters[1][i]
                etai = self.HystereticParameters[2][i]
                DeltaCi = self.HystereticParameters[3][i]
                s1p = Vyi / A[i] / E  # yield stress
                e1p = s1p / E    # yield strain
                s2p = s1p * betai
                e2p = e1p + (s2p-s1p) / (etai * E)
                s3p = s2p*1.001
                e3p = DeltaCi/storyLength
                if e3p < e2p:
                    print('WARNING: the drift of complete damage is smaller than ultimate drift')
                    e2p = e3p
                    s2p = (e2p - e1p)*(etai * E) + s1p
                    s3p = s2p*1.001
                    e3p = e2p*1.1
                if self.HystereticCurveType == 'Modified-Clough':
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, 0.5, 0.5, 
                        0, 0, 0.0)
                elif self.HystereticCurveType == 'Kinematic hardening':
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, 0.001, 0.999, 
                        0.0, 0.0, 0.0)
                elif self.HystereticCurveType == 'Pinching':
                    tao = self.HystereticParameters[4]
                    if tao == 0:
                        tao = 0.001
                    elif tao == 1:
                        tao = 0.999
                    else:
                        pass
                    py = tao
                    px = 1.0 - py
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, px, py, 
                        0, 0, 0.0)
                
                if (self.SelfCenteringEnhancingFactor > 0) & (self.SelfCenteringEnhancingFactor <= 1):
                    matTag_MultiLinear = 1000+matTag[i]
                    uniaxialMaterial('ElasticMultiLinear', matTag_MultiLinear, 
                        0.0, '-strain', -e3p,-e2p,-e1p,e1p,e2p,e3p, 
                        '-stress', -s3p,-s2p,-s1p,s1p,s2p,s3p)
                    matTag_Parallel = 2000+matTag[i]
                    uniaxialMaterial('Parallel', matTag_Parallel, matTag[i], matTag_MultiLinear, 
                        '-factors', 1.0-self.SelfCenteringEnhancingFactor,self.SelfCenteringEnhancingFactor)
            else:
                print('Error: incorrect Hysteretic Curve Type')
                return

        # element
        for i in range(self.NStories):
            if (self.SelfCenteringEnhancingFactor > 0) & (self.SelfCenteringEnhancingFactor <= 1):
                element('Truss', i+1, i,i+1, A[i], 2000+matTag[i])
            else:
                element('Truss', i+1, i,i+1, A[i], matTag[i])

        # Eigenvalue Analysis   
        if self.NStories>1:  
            lambdaN = eigen('-fullGenLapack', 2)
            w1 = lambdaN[0]**0.5
            w2 = lambdaN[1]**0.5
            T1 =  2.0*pi/w1
            T2 =  2.0*pi/w2
            if ifprint:
                print(f'Eigen Analysis: T1 = {T1:.2f} s; T2 = {T2:.2f} s')
        else:
            lambdaN = eigen('-fullGenLapack', 1)
            w1 = lambdaN[0]**0.5
            T1 =  2.0*pi/w1
            if ifprint:
                print(f'Eigen Analysis: T1 = {T1:.2f} s')

        # define & apply damping
        # RAYLEIGH damping parameters, Where to put M/K-prop damping, switches 
        # (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/1099.htm)
        # D=$alphaM*M + $betaKcurr*Kcurrent + $betaKcomm*KlastCommit + $beatKinit*$Kinitial
        if self.NStories>1: 
            xDamp = self.DampingRatio;  
            MpropSwitch = 1.0
            KcurrSwitch = 0.0
            KcommSwitch = 0.0
            KinitSwitch = 1.0
            nEigenI = 1 
            nEigenJ = 2 
            lambdaI = lambdaN[nEigenI-1] 
            lambdaJ = lambdaN[nEigenJ-1] 
            omegaI = lambdaI**0.5
            omegaJ = lambdaJ**0.5
            alphaM = MpropSwitch*xDamp*(2.0*omegaI*omegaJ)/(omegaI+omegaJ)
            betaKcurr = KcurrSwitch*2.*xDamp/(omegaI+omegaJ)      # current-K;      +beatKcurr*KCurrent
            betaKcomm = KcommSwitch*2.*xDamp/(omegaI+omegaJ)      # last-committed K;   +betaKcomm*KlastCommitt
            betaKinit = KinitSwitch*2.*xDamp/(omegaI+omegaJ)      # initial-K;     +beatKinit*Kini
            rayleigh(alphaM,betaKcurr, betaKinit, betaKcomm)       
        else:
            xDamp = self.DampingRatio;  
            MpropSwitch = 1.0
            nEigenI = 1 
            lambdaI = lambdaN[nEigenI-1] 
            omegaI = lambdaI**0.5
            alphaM = MpropSwitch*xDamp*2.0*omegaI
            rayleigh(alphaM, 0, 0, 0)  

    def __ReadDynamicRecorderFiles(self):

        # check if analysis results are empty
        fpath = str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxDrift.txt'))
        if not (os.path.isfile(fpath) and os.path.getsize(fpath) > 0):
            return

        self.MaxDrift = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxDrift.txt')), 
            sep=r'\s+', header=None).loc[2,:].values
        self.MaxAbsAccel = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxAbsAccel.txt')), 
            sep=r'\s+', header=None).loc[2,:].values
        self.MaxRelativeAccel = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxRelativeAccel.txt')), 
            sep=r'\s+', header=None).loc[2,:].values
        self.MaxAbsVel = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxAbsVel.txt')),
            sep=r'\s+', header=None).loc[2,:].values
        
        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')), 
            sep=r'\s+', header=None)
        self.DriftHistory = {}
        self.DriftHistory['time'] = df.loc[:,0]
        ind_last5sec = ((self.DriftHistory['time'][-1:]-self.DriftHistory['time'])<5.0)
        ResDrift_dict = {}
        for i in range(self.NStories):
            self.DriftHistory[i+1] = df.loc[:,i+1]
            ResDrift_dict[i+1] =  self.DriftHistory[i+1][ind_last5sec].mean()

        self.ResDrift = np.abs(np.array(list(ResDrift_dict.values()))).max()

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), 
            sep=r'\s+', header=None)
        self.ForceHistory = {}
        self.ForceHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.ForceHistory[i+1] = df.loc[:,i+1]
        
        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'NodeAbsAccelHistory.txt')), 
            sep=r'\s+', header=None)
        self.NodeAbsAccelHistory = {}
        self.NodeAbsAccelHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeAbsAccelHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'NodeRelativeAccelHistory.txt')), 
            sep=r'\s+', header=None)
        self.NodeRelativeAccelHistory = {}
        self.NodeRelativeAccelHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeRelativeAccelHistory[i+1] = df.loc[:,i+1]

    def __ReadPushoverRecorderFiles(self):

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')), 
            sep=r'\s+', header=None)
        self.DriftHistory = {}
        self.DriftHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.DriftHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), 
            sep=r'\s+', header=None)
        self.ForceHistory = {}
        self.ForceHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.ForceHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'NodeDispHistory.txt')), 
            sep=r'\s+', header=None)
        self.NodeDispHistory = {}
        self.NodeDispHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeDispHistory[i+1] = df.loc[:,i+1]