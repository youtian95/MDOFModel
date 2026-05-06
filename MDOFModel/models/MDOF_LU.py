########################################################
# 根据建筑基本信息生成结构参数。
#
# 注意：所有建筑均默认采用中等水准抗震设计
########################################################

from pathlib import Path
import numpy as np
import pandas as pd

class MDOF_LU:

    # 私有属性
    __FloorUnitMass = 1200  # 1200 kg/m2
    __SeismicDesignLevel = 'moderate-code' # 'high-code', 'moderate-code', 'low-code'
    __EQDuration = 'Moderate'  # 'Short' 'Moderate' 'Long'
    
    # 输入参数
    NumOfStories = 0
    FloorArea = 0   # m2
    StructuralType = 'UNKNOWN' # Hazus 表 5.1

    # 输出参数
    # 基本参数
    mass = 0    # kg
    K0 = 0      # N/m
    T1 = 0      # s
    # T2 = 0      # s
    N = 0
    DampingRatio = 0.05 # 阻尼比
    TypicalStoryHeight = 0 # (m)
    # 设计地震力系数
    Cs = 0
    # 骨架曲线参数
    Vdi = []    # 设计强度，单位: N
    Vyi = []    # N
    betai = [] # overstrength ratio. Utlmate strength divided by yield strength
    etai = [] # hardening ratio. post-yield stiffness divided by initial elastic stiffness
    DeltaCi = [] # ultimate drift, meter
    # hysteretic parameters
    tao = []
    # ['Modified-Clough','Kinematic hardening','Pinching']
    HystereticCurveType = 'Modified-Clough' 

    def __init__(self, NumOfStories, FloorArea, StructuralType, SeismicDesignLevel = 'UNKNOWN'):
        self.N = NumOfStories
        self.NumOfStories = NumOfStories
        self.FloorArea = FloorArea
        self.__Read_StructuralType(StructuralType)
        if SeismicDesignLevel != 'UNKNOWN':
            self.__SeismicDesignLevel = SeismicDesignLevel
        self.__Update_DesignLevel()

        # 读取 Hazus 数据
        current_directory = Path(__file__).resolve().parent.parent
        HazusDataTable5_5 = pd.read_csv(current_directory/"Resources/HazusData Table 5.5.csv",
            index_col='building type')
        HazusDataTable5_1 = pd.read_csv(current_directory/"Resources/HazusData Table 5.1.csv",
            index_col='building type')
        HazusDataTable5_4 = pd.read_csv(current_directory/"Resources/HazusData Table 5.4.csv",
            index_col='building type')
        HazusDataTable5_6 = pd.read_csv(current_directory/"Resources/HazusData Table 5.6.csv",
            index_col='building type')
        HazusDataTable5_9 = pd.read_csv(current_directory/"Resources/HazusData Table 5.9.csv",
            index_col=0, header=[0,1,2,3])
        HazusDataTable5_18 = pd.read_csv(current_directory/"Resources/HazusData Table 5.18.csv",
            index_col=0, header=[0,1])

        # 层质量
        self.mass = self.__FloorUnitMass * self.FloorArea

        # 周期
        T0 = HazusDataTable5_5['typical periods, Te (seconds)'][self.StructuralType]
        N0 = HazusDataTable5_1['typical stories'][self.StructuralType]
        self.T1 = self.N / N0 * T0
        # self.T2 = self.T1/3.0

        # 弹性层刚度
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

        # 阻尼比
        if self.StructuralType[0] == 'C': # 混凝土
            self.DampingRatio = 0.07
        elif self.StructuralType[0] == 'S': # 钢结构
            self.DampingRatio = 0.05
        elif self.StructuralType[0] == 'W': # 木结构
            self.DampingRatio = 0.10
        elif self.StructuralType[0:2] == 'RM' or self.StructuralType[0:3] == 'URM': 
            # 配筋砖石或无筋砖石
            self.DampingRatio = 0.10
        else:
            pass

        # Vyi, betai, etai
        Cs = HazusDataTable5_4[self.__SeismicDesignLevel][self.StructuralType]
        self.Cs = Cs
        gamma = HazusDataTable5_5['overstrength ratio, yield, gamma'][self.StructuralType]
        lambda_ = HazusDataTable5_5['overstrength ratio, ultimate, lambda'][self.StructuralType]
        alpha1 = HazusDataTable5_5['modal factor, weight, alpha1'][self.StructuralType]
        miu = HazusDataTable5_6[self.__SeismicDesignLevel][self.StructuralType]
        SAy = Cs*gamma/alpha1
        SAu = lambda_ * SAy
        SDy = self.mass * SAy / self.K0
        SDu = SDy * lambda_ * miu
        ISDR_threshold = HazusDataTable5_9.loc[self.StructuralType,
            (self.__SeismicDesignLevel,'Interstory Drift at Threshold of Damage State','Median','Complete')]
        kappa = HazusDataTable5_18.loc[self.StructuralType,
            (self.__SeismicDesignLevel,self.__EQDuration)]
        Height_feet = HazusDataTable5_1['typical height to roof (feet)'][self.StructuralType]
        StoryHeight = Height_feet/N0*0.3048
        self.TypicalStoryHeight = StoryHeight
        self.Vyi = [0] * self.N
        self.Vdi = [0] * self.N
        self.betai = [0] * self.N
        self.etai = [0] * self.N
        self.DeltaCi = [0] * self.N
        for i in range(self.N):
            # 注意i从0开始
            Gammai = 1.0 - (i+1.0)*i/(self.N+1.0)/self.N
            self.Vyi[i] = SAy*alpha1*self.mass*9.8*self.N*Gammai
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
        self.__SeismicDesignLevel = DesignLevel
        self.__init__(self.NumOfStories,self.FloorArea,self.StructuralType)

    def OutputStructuralParameters(self, filename):
        if isinstance(filename, str):
            if not filename.endswith('.csv'):
                filename = filename + '.csv'
            filename = Path(filename)

        data = {
            'damping ratio': [self.DampingRatio],
            'Hysteretic curve type': [self.HystereticCurveType],
            'Hysteretic parameter, tao': [self.tao],
            'Typical story height (m)': [self.TypicalStoryHeight],
            'T1 (s)': self.T1,
            'Cs': self.Cs
        }
        pd.DataFrame(data).to_csv(filename,index=0,sep=',')

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
        pd.DataFrame(data).to_csv(filename,index=0,sep=',',mode='a')

    def getDesignLevel(self):
        return self.__SeismicDesignLevel

    def __Read_StructuralType(self,StructuralType):
        current_directory = Path(__file__).resolve().parent.parent
        HazusInventoryTable4_2 = pd.read_csv(current_directory/"Resources/HazusInventory Table 4-2.csv",
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

    def __Update_DesignLevel(self):
        current_directory = Path(__file__).resolve().parent.parent
        HazusDataTable5_4 = pd.read_csv(current_directory / "Resources/HazusData Table 5.4.csv",
            index_col='building type')
        Cs = HazusDataTable5_4[self.__SeismicDesignLevel][self.StructuralType]
        if pd.isna(Cs):
            print('WARNING: Seismic design level for this building cannot be ' + 
                self.__SeismicDesignLevel + '! It is modified as lower level.')
            j_col = np.nonzero(~(HazusDataTable5_4.loc[self.StructuralType,:]
                .isna().to_numpy()))[0][0]
            self.__SeismicDesignLevel = HazusDataTable5_4.columns[j_col]

        
