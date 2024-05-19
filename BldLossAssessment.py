########################################################
# Perform building seismic loss assessment according to Hazus. Return the 
# damage states, economic losses, repair time, loss of function, etc. given 
# the EDP of buildings.
# 
# Dependancy: 
# - pandas, numpy
########################################################

from pathlib import Path
from operator import index
import numpy as np
import random
from numpy import log
import pandas as pd
import statistics as sta

class BldLossAssessment:

    __DS_type = ['Slight', 'Moderate', 'Extensive', 'Complete']

    # input parameters
    NumOfStories = 0
    FloorArea = 0   # m2
    StructuralType = 'UNKNOWN' # Hazus table 5.1
    SeismicDesignLevel = 'moderate-code' # 'high-code', 'moderate-code', 'low-code',
    OccupancyClass = 'UNKNOWN'

    ## Estimated results
    # damage states
    DS_Struct = ['UNKNOWN'] # None/ Slight/ Moderate/ Extensive/ Complete
    DS_NonStruct_DriftSen = ['UNKNOWN']
    DS_NonStruct_AccelSen = ['UNKNOWN']
    # repair cost
    RepairCost_Total = ['UNKNOWN']
    RepairCost_Struct = ['UNKNOWN']
    RepairCost_NonStruct_DriftSen = ['UNKNOWN']
    RepairCost_NonStruct_AccelSen = ['UNKNOWN']
    # repair time
    RepairTime = ['UNKNOWN'] 
    RecoveryTime = ['UNKNOWN'] # longer than repair time. It considers other factors apart from repairing components
    FunctionLossTime = ['UNKNOWN']

    ## parameters on resisual drift
    Median_RIDR = 0.01  # irrepairable residual drift ratio. 0 means not to consider it
    Beta_RIDR = 0.3

    ## Data from hazus
    # IDR/Accel thresholds for structural/Nonstructural DS
    Median_IDR_Struct_DS = [0,0,0,0]
    Beta_IDR_Struct_DS = [0,0,0,0]
    Median_IDR_NonStruct_DS = [0,0,0,0]
    Beta_IDR_NonStruct_DS = [0,0,0,0]
    Median_Accel_NonStruct_DS = [0,0,0,0]  # Unit: g
    Beta_Accel_NonStruct_DS = [0,0,0,0] # Unit: g
    # replacement cost
    ReplacementCost_Total = 0 
    StructureReplacementCost = 0  # including (1) structural, (2) drift-sens, and (3) accel-sens non-struct
    ContentsValueFactorOfStructureValue = 1.0   # accel-sensitive non-structural components
    # repair cost ratios
    StructureRCRatio_DS = [0,0,0,0] # corresponding to 4 damage states
    AccelSenNonstructRCRatio_DS = [0,0,0,0] # corresponding to 4 damage states
    DriftSenNonstructRCRatio_DS = [0,0,0,0] # corresponding to 4 damage states
    ContentsRCRatio_DS = [0,0,0,0] # corresponding to 4 accel-sensitive damage states
    # repair time
    RepairTime_DS = [0,0,0,0,0]  # corresponding to 5 damage states
    RecoveryTime_DS = [0,0,0,0,0]  # corresponding to 5 damage states
    FunctionLossMultipliers = [0,0,0,0,0]  # corresponding to 5 damage states
 
    def __init__(self, NumOfStories, FloorArea, StructuralType, DesignLevel, OccupancyClass):

        self.NumOfStories = NumOfStories
        self.FloorArea = FloorArea
        self.__Read_StructuralType(StructuralType)
        self.SeismicDesignLevel = DesignLevel
        self.OccupancyClass = OccupancyClass

        if self.OccupancyClass=='RES3':
            ind = (np.abs(np.array([2200,4400,8000,15000,40000,80000])/3.28/3.28-self.FloorArea)).argmin()
            self.OccupancyClass = self.OccupancyClass + ['A','B','C','D','E','F'][ind]

        self.__Read_StructureReplacementCost()
        self.__Read_ContentsValueFactor()
        self.ReplacementCost_Total = self.StructureReplacementCost* \
            (1.0+self.ContentsValueFactorOfStructureValue)
        self.__Read_RepairCostRatios()
        self.__Read_RepairTime_DS()
        self.__Read_IDR_Accel_thresholds_DS()

    def LossAssessment(self,MaxDriftRatio,MaxAbsAccel, MaxRIDR = 'none'):
        # Parameters:
        # MaxDriftRatio - max IDR. List[] . It is a vector if there are multiple analyses.
        # MaxAbsAccel - max AbsAccel (g). list[]. 
        # MaxRIDR - max residual drift ratio. list[].

        if len(MaxDriftRatio)==0 or len(MaxAbsAccel)==0:
            return

        self.__Estimate_DamageState(MaxDriftRatio,MaxAbsAccel,MaxRIDR)
        self.__Estimate_RepairCost()
        self.__Estimate_RepairTime()

    def __Read_StructuralType(self,StructuralType):
        current_dir = Path(__file__).resolve().parent
        HazusInventoryTable4_2 = pd.read_csv(current_dir/"Resources/HazusInventory Table 4-2.csv",
            index_col=0, header=0)
        rownames = HazusInventoryTable4_2.index.to_list()
        rownames_NO_LMH = rownames.copy()
        for i in range(0,len(rownames)):
            if rownames[i][-1] in 'LMH':
                rownames_NO_LMH[i] = rownames[i][:-1]

        if StructuralType in rownames:
            self.StructuralType = StructuralType
        elif StructuralType in rownames_NO_LMH:
            ind = [i for i in range(0,len(rownames_NO_LMH)) if StructuralType==rownames_NO_LMH[i]]
            storyrange = HazusInventoryTable4_2.iloc[ind]['story range'].values.tolist()
            for i in range(0,len(storyrange)):
                if '~' in storyrange[i]:
                    Story_low = int(storyrange[i].split('~')[0])
                    Story_high = int(storyrange[i].split('~')[1])
                elif storyrange[i]=='all':
                    Story_low = 1
                    Story_high = float('inf')
                elif '+' in storyrange[i]:
                    Story_low = int(storyrange[i][:-1])
                    Story_high = float('inf')
                else:
                    Story_low = int(storyrange[i])
                    Story_high = int(storyrange[i])
                if self.NumOfStories>=Story_low and self.NumOfStories<=Story_high:
                    self.StructuralType = rownames[ind[i]]
                    break

        else:
            self.StructuralType = StructuralType + ' is UNKNOWN'

    def __Read_StructureReplacementCost(self):
        current_dir = Path(__file__).resolve().parent
        HazusInventoryTable6_2 = pd.read_csv(
            current_dir/"./Resources/HazusInventory Table 6-2.csv",
            index_col=0, header=1)
        if self.OccupancyClass=='RES1':
            HazusInventoryTable6_3 = pd.read_csv(
                current_dir/"./Resources/HazusInventory Table 6-3.csv",
                index_col=[0,1], header=1)
            N_story = self.NumOfStories if self.NumOfStories<=3 else 3
            HeightClass = ['One-story','Two-story','Three-story'][N_story-1]
            RCPersqft = HazusInventoryTable6_3.loc[('Average',HeightClass),'Average Base cost per sq.ft'] 
        else:
            RCPersqft = HazusInventoryTable6_2.loc[self.OccupancyClass,'Structure Replacement Costl/sq.ft (2018)']
        
        assert RCPersqft[0]=='$'
        RCPersqft = float(RCPersqft[1:])
        self.StructureReplacementCost = RCPersqft*(self.FloorArea*3.28*3.28)

    def __Read_ContentsValueFactor(self):
        current_dir = Path(__file__).resolve().parent
        HazusInventoryTable6_9 = pd.read_csv(current_dir/"./Resources/HazusInventory Table 6-9.csv",
            index_col=0, header=1)
        ContentsValueFactor = HazusInventoryTable6_9.loc[self.OccupancyClass,'Contents Value (%)']
        assert ContentsValueFactor[-1:]=='%'
        self.ContentsValueFactorOfStructureValue = float(ContentsValueFactor[:-1])/100.0

    def __Read_RepairCostRatios(self):
        current_dir = Path(__file__).resolve().parent
        HazusTable15_2 = pd.read_csv(current_dir/"./Resources/HazusData Table 15.2.csv",
            index_col=1, header=2)
        HazusTable15_2 = HazusTable15_2.drop(['No.'], axis=1)
        HazusTable15_3 = pd.read_csv(current_dir/"./Resources/HazusData Table 15.3.csv",
            index_col=1, header=2)
        HazusTable15_3 = HazusTable15_3.drop(['No.'], axis=1)
        HazusTable15_4 = pd.read_csv(current_dir/"./Resources/HazusData Table 15.4.csv",
            index_col=1, header=2)
        HazusTable15_4 = HazusTable15_4.drop(['No.'], axis=1)
        HazusTable15_5 = pd.read_csv(current_dir/"./Resources/HazusData Table 15.5.csv",
            index_col=1, header=2)
        HazusTable15_5 = HazusTable15_5.drop(['No.'], axis=1)
        self.StructureRCRatio_DS = (HazusTable15_2.loc[self.OccupancyClass].values/100.0).tolist()
        self.AccelSenNonstructRCRatio_DS = (HazusTable15_3.loc[self.OccupancyClass].values/100.0).tolist()
        self.DriftSenNonstructRCRatio_DS = (HazusTable15_4.loc[self.OccupancyClass].values/100.0).tolist()
        self.ContentsRCRatio_DS = (HazusTable15_5.loc[self.OccupancyClass].values/100.0).tolist()

    def __Read_RepairTime_DS(self):
        current_dir = Path(__file__).resolve().parent
        HazusData4_2_Table11_7 = pd.read_csv(current_dir/"./Resources/HazusData4-2 Table 11-7.csv",
            index_col=1, header=2)
        HazusData4_2_Table11_7 = HazusData4_2_Table11_7.drop(['No.'], axis=1)
        HazusData4_2_Table11_8 = pd.read_csv(current_dir/"./Resources/HazusData4-2 Table 11-8.csv",
            index_col=1, header=2)
        HazusData4_2_Table11_8 = HazusData4_2_Table11_8.drop(['No.'], axis=1)
        HazusData4_2_Table11_9 = pd.read_csv(current_dir/"./Resources/HazusData4-2 Table 11-9.csv",
            index_col=1, header=2)
        HazusData4_2_Table11_9 = HazusData4_2_Table11_9.drop(['No.'], axis=1)
        self.RepairTime_DS = HazusData4_2_Table11_7.loc[self.OccupancyClass].values.tolist()
        self.RecoveryTime_DS = HazusData4_2_Table11_8.loc[self.OccupancyClass].values.tolist()
        self.FunctionLossMultipliers = HazusData4_2_Table11_9.loc[self.OccupancyClass].values.tolist()
        
    def __Read_IDR_Accel_thresholds_DS(self):
        current_dir = Path(__file__).resolve().parent
        HazusTable5_9 = pd.read_csv(current_dir/"./Resources/HazusData Table 5.9.csv",
            index_col=0, header=[0,1,2,3])
        HazusTable5_10 = pd.read_csv(current_dir/"./Resources/HazusData Table 5.10.csv",
            index_col=None, header=[1,2])
        HazusTable5_12 = pd.read_csv(current_dir/"./Resources/HazusData Table 5.12.csv",
            index_col=0, header=[1,2])
        self.Median_IDR_Struct_DS = HazusTable5_9.loc[self.StructuralType,(self.SeismicDesignLevel,
            'Interstory Drift at Threshold of Damage State','Median')].values.tolist()
        self.Beta_IDR_Struct_DS = HazusTable5_9.loc[self.StructuralType,(self.SeismicDesignLevel,
            'Interstory Drift at Threshold of Damage State','Beta')].values.tolist()
        self.Median_IDR_NonStruct_DS = HazusTable5_10.loc[0,('Median')].values.tolist()
        self.Beta_IDR_NonStruct_DS = HazusTable5_10.loc[0,('Beta')].values.tolist()
        self.Median_Accel_NonStruct_DS = HazusTable5_12.loc[self.SeismicDesignLevel,('Median')].values.tolist()
        self.Beta_Accel_NonStruct_DS = HazusTable5_12.loc[self.SeismicDesignLevel,('Beta')].values.tolist()
        
    def __Estimate_DamageState(self,MaxDriftRatio,MaxAbsAccel,MaxRIDR):

        # normal distribution objects
        nd_DS_Struct = []
        for a, b in zip(self.Median_IDR_Struct_DS,self.Beta_IDR_Struct_DS):
            nd_DS_Struct.append(sta.NormalDist(log(a),b))
        nd_DS_NonStruct_Drift = []
        for a, b in zip(self.Median_IDR_NonStruct_DS,self.Beta_IDR_NonStruct_DS):
            nd_DS_NonStruct_Drift.append(sta.NormalDist(log(a),b))
        nd_DS_NonStruct_Accel = []
        for a, b in zip(self.Median_Accel_NonStruct_DS,self.Beta_Accel_NonStruct_DS):
            nd_DS_NonStruct_Accel.append(sta.NormalDist(log(a),b))
        
        if not ((self.Median_RIDR ==0) or (MaxRIDR=='none')):
            nd_irrepairable = sta.NormalDist(log(self.Median_RIDR),self.Beta_RIDR)

        self.DS_Struct = ['None'] * len(MaxDriftRatio)
        self.DS_NonStruct_DriftSen = ['None']* len(MaxDriftRatio)
        self.DS_NonStruct_AccelSen = ['None']* len(MaxDriftRatio)
        i = 0
        for d, a in zip(MaxDriftRatio, MaxAbsAccel):
            # damage states

            # irrepairable
            if not ((self.Median_RIDR ==0) or (MaxRIDR=='none')):
                assert isinstance(MaxRIDR,list)
                assert len(MaxRIDR)==len(MaxDriftRatio)

                P_irrepairable = nd_irrepairable.cdf(log(MaxRIDR[i]))
                if random.random()<=P_irrepairable:
                    # irrepairable
                    self.DS_Struct[i] = self.__DS_type[-1]
                    self.DS_NonStruct_DriftSen[i] = self.__DS_type[-1]
                    self.DS_NonStruct_AccelSen[i] = self.__DS_type[-1]

                    i+=1
                    continue

            # repairable
            P_DS_Struct = [nd.cdf(log(d)) for nd in nd_DS_Struct]
            P_DS_NonStruct_Drift = [nd.cdf(log(d)) for nd in nd_DS_NonStruct_Drift]
            P_DS_NonStruct_Accel = [nd.cdf(log(a)) for nd in nd_DS_NonStruct_Accel]

            ind = np.nonzero(np.array(P_DS_Struct)>=random.random())[0]
            if ind.size>0:
                self.DS_Struct[i] = self.__DS_type[ind[-1]]

            ind = np.nonzero(np.array(P_DS_NonStruct_Drift)>=random.random())[0]
            if ind.size>0:
                self.DS_NonStruct_DriftSen[i] = self.__DS_type[ind[-1]]

            ind = np.nonzero(np.array(P_DS_NonStruct_Accel)>=random.random())[0]
            if ind.size>0:
                self.DS_NonStruct_AccelSen[i] = self.__DS_type[ind[-1]]

            i+=1

    def __Estimate_RepairCost(self):
        # given Damage States
        self.RepairCost_Struct = [0]*len(self.DS_Struct)
        self.RepairCost_NonStruct_DriftSen = [0]*len(self.DS_Struct)
        self.RepairCost_NonStruct_AccelSen = [0]*len(self.DS_Struct)
        self.RepairCost_Total = [0]*len(self.DS_Struct)
        i=0
        for ds1,ds2,ds3 in zip(self.DS_Struct,self.DS_NonStruct_DriftSen,self.DS_NonStruct_AccelSen):

            if ds1 in self.__DS_type:
                ind = self.__DS_type.index(ds1) 
                self.RepairCost_Struct[i] = self.StructureRCRatio_DS[ind]*self.StructureReplacementCost
            else:
                self.RepairCost_Struct[i] = 0

            if ds2 in self.__DS_type:
                ind = self.__DS_type.index(ds2) 
                self.RepairCost_NonStruct_DriftSen[i] = \
                    self.DriftSenNonstructRCRatio_DS[ind]*self.StructureReplacementCost
            else:
                self.RepairCost_NonStruct_DriftSen[i] = 0

            if ds3 in self.__DS_type:
                ind = self.__DS_type.index(ds3) 
                self.RepairCost_NonStruct_AccelSen[i] = \
                    self.AccelSenNonstructRCRatio_DS[ind]*self.StructureReplacementCost \
                    + self.ContentsRCRatio_DS[ind]*self.StructureReplacementCost
            else:
                self.RepairCost_NonStruct_AccelSen[i] = 0

            self.RepairCost_Total[i] = self.RepairCost_Struct[i] + \
                self.RepairCost_NonStruct_DriftSen[i] + \
                self.RepairCost_NonStruct_AccelSen[i]

            i+=1

    def __Estimate_RepairTime(self):
        self.RepairTime = [0]*len(self.DS_Struct)
        self.RecoveryTime = [0]*len(self.DS_Struct)
        self.FunctionLossTime = [0]*len(self.DS_Struct)

        i = 0
        for ds in self.DS_Struct:
            ind = self.__DS_type.index(ds) if ds in self.__DS_type else -1
            self.RepairTime[i] = self.RepairTime_DS[ind+1]
            self.RecoveryTime[i] = self.RecoveryTime_DS[ind+1]
            self.FunctionLossTime[i] = self.RecoveryTime[i]*self.FunctionLossMultipliers[ind+1]
            i+=1
