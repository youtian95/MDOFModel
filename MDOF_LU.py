########################################################
# Generate structural parameters according to basic building information.
# 
# Note: all buildings are considered as moderate seismic design level
# 
# Dependancy: 
# - numpy, pandas
########################################################

import numpy as np
import pandas as pd

class MDOF_LU:

    # private
    __FloorUnitMass = 1200  # 1200 kg/m2
    __SeismicDesignLevel = 'moderate-code'
    
    # input parameters
    NumOfStories = 0
    FloorArea = 0   # m2
    StructuralType = 'UNKNOWN' # Hazus table 5.1

    # output parameters
    # basic
    mass = 0    # kg
    K0 = 0      # N/m
    T1 = 0      # s
    T2 = 0      # s
    N = 0
    eta = 0.05 # damping ratio
    # backbone curve
    Vyi = []    # N
    betai = [] # overstrength ratio. Utlmate strength divided by yield strength
    etai = [] # hardening ratio
    DeltaCi = [] # ultimate drift, meter


    def __init__(self, NumOfStories, FloorArea, StructuralType):
        self.N = NumOfStories
        self.NumOfStories = NumOfStories
        self.FloorArea = FloorArea
        self.StructuralType = StructuralType

        # read hazus data
        HazusDataTable5_5 = pd.read_csv("./Resources/HazusData Table 5.5.csv",
            index_col='building type')
        HazusDataTable5_1 = pd.read_csv("./Resources/HazusData Table 5.1.csv",
            index_col='building type')
        HazusDataTable5_4 = pd.read_csv("./Resources/HazusData Table 5.4.csv",
            index_col='building type')
        HazusDataTable5_6 = pd.read_csv("./Resources/HazusData Table 5.6.csv",
            index_col='building type')
        HazusDataTable5_9 = pd.read_csv("./Resources/HazusData Table 5.9.csv",
            index_col=0, header=[0,1,2])

        # story mass
        self.mass = self.__FloorUnitMass * self.FloorArea

        # periods
        T0 = HazusDataTable5_5['typical periods, Te (seconds)'][self.StructuralType]
        N0 = HazusDataTable5_1['typical stories'][self.StructuralType]
        self.T1 = self.N / N0 * T0
        self.T2 = self.T1/3.0

        # elastic stiffness
        UnitMassMat = np.zeros([self.N,self.N])
        if self.N == 1:
            lambda1 = 1
        elif self.N > 1:
            for i in range(0,self.N-1):
                UnitMassMat[i,i] = 2
                UnitMassMat[i,i+1] = -1
            for i in range(1,self.N):
                UnitMassMat[i,i-1] = -1
            MassMat = self.mass * UnitMassMat
            lambda1, featurevector = np.linalg.eig(MassMat)
        else:
            pass
        self.K0 = 4.0*3.14**2*self.mass/self.T1**2/lambda1

        # damping ratio
        if self.StructuralType[0] == 'C': # concrete
            self.eta = 0.07
        elif self.StructuralType[0] == 'S': # steel
            self.eta = 0.05
        elif self.StructuralType[0] == 'W': # wood
            self.eta = 0.10
        elif self.StructuralType[0:2] == 'RM' or self.StructuralType[0:3] == 'URM': 
            # reinforced mansory or unreinforced mansory
            self.eta = 0.10
        else:
            pass

        # Vyi, betai, etai
        Cs = HazusDataTable5_4[self.__SeismicDesignLevel][self.StructuralType]
        gamma = HazusDataTable5_4['overstrength ratio, yield, gamma'][self.StructuralType]
        lambda_ = HazusDataTable5_4['overstrength ratio, ultimate, lambda'][self.StructuralType]
        alpha1 = HazusDataTable5_4['modal factor, weight, alpha1'][self.StructuralType]
        miu = HazusDataTable5_6[self.__SeismicDesignLevel][self.StructuralType]
        SAy = Cs*gamma/alpha1
        SAu = lambda_ * SAy
        SDy = self.mass * SAy / self.K0
        SDu = SDy * lambda_ * miu
        ISDR_threshold = HazusDataTable5_9.loc[self.StructuralType,
            (self.__SeismicDesignLevel,'Interstory Drift at Threshold of Damage State','-','Complete')]
        Height_feet = HazusDataTable5_1['typical height to roof (feet)'][self.StructuralType]
        StoryHeight = Height_feet/N0*0.3048
        self.Vyi = [0] * self.N
        self.betai = [0] * self.N
        self.etai = [0] * self.N
        self.DeltaCi = [0] * self.N
        for i in range(self.N):
            Gammai = 1.0 - i*(i-1.0)/(self.N+1.0)/self.N
            self.Vyi[i] = SAy*alpha1*self.mass*9.8*self.N*Gammai
            self.betai[i] = SAu / SAy
            self.etai[i] = (SAu - SAy) / (SDu - SDy) * SDy / SAy
            self.DeltaCi[i] = StoryHeight*ISDR_threshold



