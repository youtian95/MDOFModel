########################################################
# Generate structural parameters according to basic building information based on CN standards.
# 
# Dependancy: 
# - numpy, pandas
########################################################

from pathlib import Path
import numpy as np
import pandas as pd
import re

import Alpha_CNcode as ACN

class MDOF_CN:

    # private
    __FloorUnitMass = 1200  # 1200 kg/m2
    
    # input parameters
    NumOfStories = 0
    FloorArea = 0   # m2
    StructuralType = 'UNKNOWN' # Hazus table 5.1
    SeismicDesignLevel = '7' # 6, 7, 7.5, 8, 8.5, 9 (g)
    EQGroup = '2' # 1, 2, 3
    SiteClass = '3'  # 1_0, 1_1, 2, 3, 4
    longitude = None
    latitude = None

    # output parameters
    # basic
    mass = 0    # kg
    K0 = 0      # N/m
    T1 = 0      # s
    N = 0
    DampingRatio = 0.05 # damping ratio
    TypicalStoryHeight = 0 # (m)
    # backbone curve
    Vdi = []    # design strength (N)  (475-year return period)
    Vyi = []    # N
    betai = [] # overstrength ratio. Utlmate strength divided by yield strength
    etai = [] # hardening ratio
    DeltaCi = [] # ultimate drift, meter
    # hysteretic parameters
    tao = []
    # ['Modified-Clough','Kinematic hardening','Pinching']
    HystereticCurveType = 'Modified-Clough' 

    # If the seismic design level or EQgroup is not provided, they will be set according to the city.
    # If the site class is not provided, it will be set according to the location.
    def __init__(self, NumOfStories, FloorArea, StructuralType, 
            SeismicDesignLevel = 'UNKNOWN', EQGroup = 'UNKNOWN', City='UNKNOWN', 
            SiteClass='UNKNOWN', longitude = None, latitude = None):
        self.N = NumOfStories
        self.NumOfStories = NumOfStories
        self.FloorArea = FloorArea
        self.__Read_StructuralType(StructuralType)
        # design level
        if not (SeismicDesignLevel == 'UNKNOWN'):
            self.SeismicDesignLevel = SeismicDesignLevel
        if not (EQGroup == 'UNKNOWN'):
            self.EQGroup = EQGroup
        if (not (City == 'UNKNOWN')) and \
            ((SeismicDesignLevel == 'UNKNOWN') or (EQGroup == 'UNKNOWN')):
            self.__Set_DesignLevelbyCity(City)
        # site class
        self.longitude = longitude
        self.latitude = latitude
        if not (SiteClass == 'UNKNOWN'):
            self.SiteClass = SiteClass
        else:
            if (longitude and latitude):
                self.__Set_SiteClassbyLoc(longitude,latitude)
            
        # story mass
        self.mass = self.__FloorUnitMass * self.FloorArea

        # read hazus data
        current_path = Path(__file__).resolve().parent
        HazusDataTable5_5 = pd.read_csv(current_path/"./Resources/HazusData Table 5.5.csv",
            index_col='building type')
        HazusDataTable5_1 = pd.read_csv(current_path/"./Resources/HazusData Table 5.1.csv",
            index_col='building type')
        HazusDataTable5_6 = pd.read_csv(current_path/"./Resources/HazusData Table 5.6.csv",
            index_col='building type')
        HazusDataTable5_9 = pd.read_csv(current_path/"./Resources/HazusData Table 5.9.csv",
            index_col=0, header=[0,1,2,3])
        HazusDataTable5_18 = pd.read_csv(current_path/"./Resources/HazusData Table 5.18.csv",
            index_col=0, header=[0,1])
        
        # Concert_CN2Hazus_SeismicDesignLevel
        SDL_Hazus = ACN.Concert_CN2Hazus_SeismicDesignLevel(self.SeismicDesignLevel)

        # periods. According to Hazus Table 5.5
        T0 = HazusDataTable5_5['typical periods, Te (seconds)'][self.StructuralType]
        N0 = HazusDataTable5_1['typical stories'][self.StructuralType]
        self.T1 = self.N / N0 * T0
        # According to CN code, if the building has more than 10 stories, the period is calculated as follows:
        # [1] 中国建筑科学研究院, 同济大学, 中国建筑设计研究院, 等. 建筑结构荷载规范（GB 50009-2012） [S]. 2012.
        if self.N >= 10:
            if self.StructuralType[0] == 'C': # concrete
                self.T1 = 0.075*self.N
            elif self.StructuralType[0] == 'S': # steel
                self.T1 = 0.125*self.N

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
            UnitMassMat[-1,-1] = 1
            lambda_list, featurevector = np.linalg.eig(UnitMassMat)
            lambda1 = lambda_list.min()
        else:
            pass
        self.K0 = 4.0*3.14**2*self.mass/self.T1**2/lambda1

        # damping ratio
        if self.StructuralType[0] == 'C': # concrete
            self.DampingRatio = 0.07
        elif self.StructuralType[0] == 'S': # steel
            self.DampingRatio = 0.05
        elif self.StructuralType[0] == 'W': # wood
            self.DampingRatio = 0.10
        elif self.StructuralType[0:2] == 'RM' or self.StructuralType[0:3] == 'URM': 
            # reinforced mansory or unreinforced mansory
            self.DampingRatio = 0.10
        else:
            pass

        # Tg
        Tg = ACN.Tg_CNcode(self.EQGroup,self.SiteClass)
        # alphaMax_medium
        alphaMax_medium = ACN.alphaMax_CNcode('medium',self.SeismicDesignLevel)
        alpha1_medium = ACN.Alpha_CNcode(self.T1,Tg,alphaMax_medium,self.DampingRatio)
        alphaMax_major = ACN.alphaMax_CNcode('major',self.SeismicDesignLevel)
        alpha1_major = ACN.Alpha_CNcode(self.T1,Tg,alphaMax_major,self.DampingRatio)

        # Vyi, betai, etai
        # Per GB 50011-2010
        kesi_y = 0.4 # Table 5.5.4, GB 50011-2010
        SAy = 0.85*alpha1_major*kesi_y
        SDy = self.mass * SAy / self.K0
        gamma = (alpha1_major*kesi_y)/alpha1_medium  # 'overstrength ratio, yield, gamma'
        lambda_ = HazusDataTable5_5['overstrength ratio, ultimate, lambda'][self.StructuralType]
        SAu = lambda_ * SAy
        miu = HazusDataTable5_6[SDL_Hazus][self.StructuralType]
        SDu = SDy * lambda_ * miu
        ISDR_threshold = HazusDataTable5_9.loc[self.StructuralType,
            (SDL_Hazus,'Interstory Drift at Threshold of Damage State','Median','Complete')]
        kappa = HazusDataTable5_18.loc[self.StructuralType,(SDL_Hazus,'Moderate')]
        # typical story height
        Height_feet = HazusDataTable5_1['typical height to roof (feet)'][self.StructuralType]
        StoryHeight = Height_feet/N0*0.3048
        self.TypicalStoryHeight = StoryHeight

        # Vyi, Vdi, betai, etai of other stories
        self.Vyi = [0] * self.N
        self.Vdi = [0] * self.N
        self.betai = [0] * self.N
        self.etai = [0] * self.N
        self.DeltaCi = [0] * self.N
        for i in range(self.N):
            Gammai = 1.0 - i*(i+1.0)/(self.N+1.0)/self.N
            self.Vyi[i] = SAy*self.mass*9.8*self.N*Gammai
            self.Vdi[i] = self.Vyi[i]/gamma
            self.betai[i] = SAu / SAy
            self.etai[i] = (SAu - SAy) / (SDu - SDy) * SDy / SAy
            self.DeltaCi[i] = StoryHeight*ISDR_threshold

        # hysteretic parameters
        if self.StructuralType[0:2] == 'C1': # concrete
            self.HystereticCurveType = 'Modified-Clough'
        elif self.StructuralType[0:2] in ['S1','S3']: # steel
            self.HystereticCurveType = 'Kinematic hardening'
        else:
            self.HystereticCurveType = 'Pinching'
            self.tao = kappa

    def set_DesignLevel(self, DesignLevel: str):
        self.SeismicDesignLevel = DesignLevel
        self.__init__(self.NumOfStories,self.FloorArea,self.StructuralType)

    def OutputStructuralParameters(self, filename):

        data = {
            'damping ratio': [self.DampingRatio],
            'Hysteretic curve type': [self.HystereticCurveType],
            'Hysteretic parameter, tao': [self.tao],
            'Typical story height (m)': [self.TypicalStoryHeight]
        }
        pd.DataFrame(data).to_csv(filename +'.csv',index=0,sep=',')

        yileddisp = np.array(self.Vyi)/self.K0
        designforce = np.array(self.Vdi)
        designdisp = designforce/self.K0
        ultforce = np.array(self.betai)*np.array(self.Vyi)
        ultdisp = yileddisp + (ultforce - np.array(self.Vyi))/(self.K0*np.array(self.etai))
        data = {
            'No. of story': list(range(1,self.N+1)), 
            'Floor mass (kg)': [self.mass]*self.N,
            'Elastic shear stiffness (N/m)': [self.K0]*self.N,
            'Design shear force (N)': self.Vdi,
            'Design displacement (m)': designdisp.tolist(),
            'Yield shear force (N)': self.Vyi,
            'Yield displacement (m)': yileddisp.tolist(),
            'Ultimate shear force (N)': ultforce.tolist(),
            'Ultimage displacement (m)': ultdisp.tolist(),
            'Complete damage displacement (m)': self.DeltaCi,
        }
        pd.DataFrame(data).to_csv(filename +'.csv',index=0,sep=',',mode='a')

    # Generate detailed structural types (like S2) according to reference [1], if only a general type (like S) is provided.
    # [1] FEMA. Hazus Inventory Technical Manual [R]. Hazus 4.2 SP3. FEMA, 2021.
    def __Read_StructuralType(self,StructuralType):
        current_path = Path(__file__).resolve().parent
        HazusInventoryTable4_2 = pd.read_csv(current_path/"./Resources/HazusInventory Table 4-2.csv",
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

    # Set seismic design level according to city
    # [1] GB 50011-2010(2016) Appendix A
    def __Set_DesignLevelbyCity(self, city: str):
        current_path = Path(__file__).resolve().parent
        GBApp_A = pd.read_csv(current_path/"./Resources/GB50011-2010(2016)-Appendix-A.csv",
            na_values='-')
        Row = GBApp_A[GBApp_A['City'].str.contains(city,na=False)]
        if Row.empty:
            print("Error: cannot find such city")
            raise SystemExit
        else:
            SDL = Row['Design Level'].values[-1]   
            SDL = re.findall(r"\d+\.?\d*", SDL)[0]
            alphaMax = Row['Alpha_max'].values[-1]
            alphaMax = re.findall(r"\d+\.?\d*", alphaMax)[0]
            alphaMax = float(alphaMax)
            if SDL == '8' and alphaMax == 0.3:
                SDL = '8.5'
            elif SDL == '7' and alphaMax == 0.15:
                SDL = '7.5'
            self.SeismicDesignLevel = SDL
            EQGroup = Row['EQgroup'].values[-1]
            if EQGroup[1] == '一':
                EQGroup = '1'
            elif EQGroup[1] == '二':
                EQGroup = '2'
            elif EQGroup[1] == '三':
                EQGroup = '3'
            self.EQGroup = EQGroup

    # Set site class according to location per GB 50011-2010(2016) Table 4.1.6
    # [1] GB 50011-2010(2016) Table 4.1.6
    # [2] Zhou J, Li X, Tian X, Xu G. New Framework of Combining Observations with Topographic Slope to Estimate VS30 and Its Application on Building a VS30 Map for Mainland China. Bulletin of the Seismological Society of America, 2022, 112(4): 2049-2069.
    def __Set_SiteClassbyLoc(self, Longitude: float, Latitude: float):
        current_path = Path(__file__).resolve().parent
        VS30Table = pd.read_excel(current_path/"./Resources/China_Mainland_SCK_Vs30.xlsx",header=1)
        distances = np.sqrt((VS30Table['Longitude (°)'] - Longitude)**2 \
            + (VS30Table['Latitude (°)'] - Latitude)**2)
        closest_index = distances.idxmin()
        VS30 = VS30Table.iloc[closest_index-1, 4]
        if VS30 > 800:
            SiteClass = '1_0'
        elif VS30 > 500:
            SiteClass = '1_1'
        elif VS30 > 250:
            SiteClass = '2'
        elif VS30 > 150:
            SiteClass = '3'
        else:
            SiteClass = '4'
        self.SiteClass = SiteClass


        

