########################################################
# Perform dynamic analysis using Openseespy 
# 
# Dependancy: 
# - openseespy, pandas, numpy, matplotlib
########################################################

import MDOF_LU as mlu
import matplotlib.pyplot as plt
from cmath import pi
from openseespy.opensees import *
import pandas as pd
import ReadRecord

class MDOFOpenSees():

    # structrual parameters
    NStories : int = 0
    m: list = []
    k: list = []
    DampingRatio:float = 0.05
    HystereticCurveType: str = 'Elastic'
    HystereticParameters = ()

    # Dynamic analysis results
    MaxDrift = [] # MaxDrift[0] is the 1st story
    MaxAbsAccel = [] # MaxAbsAccel[0] is the ground
    MaxRelativeAccel = [] # [0] is the ground
    ResDrift = 0.0
    DriftHistory = {} # DriftHistory['time'] is the time list. DriftHistory[1] is the IDR list of 1st story
    ForceHistory = {} 
    NodeAbsAccelHistory = {} # NodeAbsAccelHistory[0] is the ground
    NodeRelativeAccelHistory = {}

    def __init__(self, NStories :int, m: list, k:list, DampingRatio:float,
        HystereticCurveType: str, *HystereticParameters):
        # -NStories, number of stories
        # -m, mass list for each floor, kg
        # -k, elastic stiffness for each story, N/m
        # -DampingRatio, scalar
        # -HystereticCurveType, ['Elastic','Modified-Clough','Kinematic hardening','Pinching']
        # -*HystereticParameters, variable parameters including (Vyi, betai, etai, DeltaCi, tao)
        # {
        #   -Vyi, yield shear force for each story, N
        #   -betai, overstrength ratio of ultimate strength to yield strength for each story
        #   -etai, hardening ratio for each story
        #   -DeltaCi, displacement threshold for complete damage state, m
        # }

        self.NStories = NStories
        self.m = m
        self.k = k
        self.DampingRatio = DampingRatio
        self.HystereticCurveType = HystereticCurveType
        self.HystereticParameters = HystereticParameters

    def DynamicAnalysis(self,ifprint: bool, EQRecordfile:str, GMScaling:float):
        # Parameters:
        # -ifprint, true or false
        # -EQRecordfile, earthquake record file which is in PEER format, such as 'H-E12140'
        # -GMScaling, ground motion scaling factor
        # 
        # Return:
        # Iffinish, tCurrent, TotalTime

        if ifprint:
            print('Perform dynamic analysis of a MDOF lumped-mass building model with OpenSees...')

        wipe()						
        model('basic', '-ndm', 2, '-ndf', 3)

        dof1 = 1
        dof2 = 2
        dof3 = 3
        g = 9.8
        floorLength = 1.0

        # node
        node(0, 0., 0.)
        fix(0, 1, 1, 1) 
        for i in range(self.NStories):
            node(i+1, (i+1)*floorLength, 0.)
            mass(i+1, self.m[i], 0., 0.)
            fix(i+1, 0, 1, 1) 
        
        # material
        E = 1.0
        matTag = [i+1 for i in range(self.NStories)]
        A = [0] * self.NStories
        for i in range(self.NStories):
            A[i] = self.k[i] * floorLength / E
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
                s3p = s2p
                e3p = DeltaCi/floorLength
                if self.HystereticCurveType == 'Modified-Clough':
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, 0.5, 0.5, 
                        0, 0, 0.0)
                elif self.HystereticCurveType == 'Kinematic hardening':
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, 0.0, 1.0, 
                        0.0, 0.0, 0.0)
                elif self.HystereticCurveType == 'Pinching':
                    tao = self.HystereticParameters[4]
                    py = tao
                    px = 1.0 - py
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, px, py, 
                        0, 0, 0.0)
            else:
                print('Error: incorrect Hysteretic Curve Type')
                return

        # element
        for i in range(self.NStories):
            element('Truss', i+1, i,i+1, A[i], matTag[i])

        # Eigenvalue Analysis     
        lambdaN = eigen('-fullGenLapack', 2)
        w1 = lambdaN[0]**0.5
        w2 = lambdaN[1]**0.5
        T1 =  2.0*pi/w1
        T2 =  2.0*pi/w2
        if ifprint:
            print(f'Eigen Analysis: T1 = {T1:.2f} s; T2 = {T2:.2f} s')

        # define & apply damping
        # RAYLEIGH damping parameters, Where to put M/K-prop damping, switches 
        # (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/1099.htm)
        # D=$alphaM*M + $betaKcurr*Kcurrent + $betaKcomm*KlastCommit + $beatKinit*$Kinitial
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
        
        # Permform the conversion from SMD record to OpenSees record
        dt, nPts = ReadRecord.ReadRecord(EQRecordfile +'.at2', EQRecordfile +'.dat')

        
        # Uniform EXCITATION: acceleration input
        tsTag = 100
        timeSeries('Path', tsTag, '-dt', dt, '-filePath', EQRecordfile +'.dat', '-factor', g * GMScaling)
        IDloadTag = 400			# load tag
        GMdirection = 1
        pattern('UniformExcitation', IDloadTag, GMdirection, '-accel', tsTag)

        # recorders
        recorder('EnvelopeElement', '-file', 'MaxDrift.txt',
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 'DriftHistory.txt', '-time',
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 'ForceHistory.txt', '-time',
            '-ele', *list(range(1,self.NStories+1)), 'axialForce')
        recorder('EnvelopeNode', '-file', 'MaxAbsAccel.txt', '-timeSeries', tsTag, 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('EnvelopeNode', '-file', 'MaxRelativeAccel.txt', 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('Node', '-file', 'NodeAbsAccelHistory.txt', '-timeSeries', tsTag, '-time', 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('Node', '-file', 'NodeRelativeAccelHistory.txt', '-time', 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')


        # dynamic analysis
        Tol = 1e-8
        maxNumIter = 10
        DtAnalysis = 0.001  # dt
        wipeAnalysis()
        constraints('Transformation')
        numberer('RCM')
        system('BandGeneral')

        tCurrent = getTime()

        Test = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 3:'EnergyIncr', 
            4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
        Algorithm = {1:'KrylovNewton', 2: 'SecantNewton' , 3:'ModifiedNewton' , 
            4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

        tFinal = nPts*dt

        time = [tCurrent]
        ok = 0
        while tCurrent < tFinal:   
            for i in Test:
                test(Test[i], Tol, maxNumIter)    
                for j in Algorithm: 
                    if j < 4:
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

        Iffinish = not ok
        TotalTime = tFinal

        if ifprint:
            print(f'State (Successful or Fault): {Iffinish:d}')
            print(f'The analysis ends at {tCurrent:.3f} sec out of {TotalTime:.3f} sec.')
        
        wipe()
        self.__ReadRecorderFiles()

        return Iffinish, tCurrent, TotalTime

    def PlotForceDriftHistory(self, NumOfStory:int = 1):
        fig, ax = plt.subplots()  
        ax.plot(self.DriftHistory[NumOfStory],self.ForceHistory[NumOfStory]); 
        plt.show()

    def __ReadRecorderFiles(self):

        self.MaxDrift = pd.read_table('MaxDrift.txt', sep='\s+', header=None).loc[2,:]
        self.MaxAbsAccel = pd.read_table('MaxAbsAccel.txt', sep='\s+', header=None).loc[2,:]
        self.MaxRelativeAccel = pd.read_table('MaxRelativeAccel.txt', sep='\s+', header=None).loc[2,:]
        
        df = pd.read_table('DriftHistory.txt', sep='\s+', header=None)
        self.DriftHistory = {}
        self.DriftHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.DriftHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table('ForceHistory.txt', sep='\s+', header=None)
        self.ForceHistory = {}
        self.ForceHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.ForceHistory[i+1] = df.loc[:,i+1]
        
        df = pd.read_table('NodeAbsAccelHistory.txt', sep='\s+', header=None)
        self.NodeAbsAccelHistory = {}
        self.NodeAbsAccelHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeAbsAccelHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table('NodeRelativeAccelHistory.txt', sep='\s+', header=None)
        self.NodeRelativeAccelHistory = {}
        self.NodeRelativeAccelHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeRelativeAccelHistory[i+1] = df.loc[:,i+1]

bld = mlu.MDOF_LU(3, 1000, 'C1M')
bld.OutputStructuralParameters('structural parameters')

fe = MDOFOpenSees(3, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi)
fe.DynamicAnalysis(1, 'H-E12140', 1.0)

fe.PlotForceDriftHistory(1)