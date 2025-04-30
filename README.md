# MDOFModel

基于Python的多自由度(MDOF)结构地震工程分析库。

## 项目介绍

MDOFModel是一个用于结构工程中多自由度(MDOF)模型分析的Python库，主要用于地震工程分析。该工具可以通过基本建筑信息生成集中质量有限元模型，进行动力分析、推覆分析、损失评估和增量动力分析(IDA)等。

## 功能特点

- **结构模型生成**：基于基本建筑参数(如层数、面积、结构类型)生成MDOF结构模型
- **地震动力分析**：进行线性和非线性时程分析
- **推覆分析**：实施结构静力推覆分析
- **增量动力分析(IDA)**：采用FEMA P-695远场地震记录执行IDA分析
- **损失评估**：基于Hazus方法进行地震损失评估
- **OpenSees集成**：与OpenSees进行无缝对接

## 安装说明

使用pip安装本库：

```bash
pip install MDOFModel
```

### 依赖项

- Python >= 3.12
- numpy
- pandas
- matplotlib
- openseespy
- openpyxl
- eqsig

## 使用示例

### 示例1：动力分析

```python
from MDOFModel import MDOF_CN as mcn
from MDOFModel import MDOFOpenSees as mops

# 创建3层结构模型
NumofStories = 3
bld = mcn.MDOF_CN(NumofStories, 1000, 'S2', City='石家庄',longitude=114.52,latitude=38.05)
bld.OutputStructuralParameters('structural parameters')

# 执行动力分析
fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.DynamicAnalysis('H-E12140', 3.0, True)

# 绘制层间位移时程
fe.PlotForceDriftHistory(1)
```

### 示例2：增量动力分析(IDA)

```python
from MDOFModel import IDA
from MDOFModel import MDOF_LU as mlu
from MDOFModel import MDOFOpenSees as mops
import numpy as np

# 创建结构模型
NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 3600, 'S2')
bld.set_DesignLevel('pre-code')

# 设置OpenSees模型
fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)

# 执行IDA分析
IM_list = np.linspace(0.1, 2.0, 10).tolist()
IDA_obj = IDA.IDA(fe)
IDA_result = IDA_obj.Analyze(IM_list, EQRecordFile_list, bld.T1)

# 保存和绘制结果
IDA_result.to_csv('IDA_results.csv')
IDA.IDA.plot_IDA_results(IDA_result, Stat=True, FigName='IDA.jpg')
```

## 主要模块说明

- **MDOF_CN**：基于中国规范的多自由度模型生成
- **MDOF_LU**：通用多自由度模型生成
- **MDOFOpenSees**：OpenSees接口，用于建模和分析
- **IDA**：增量动力分析
- **BldLossAssessment**：建筑损失评估
- **Tool_IDA**：IDA分析辅助工具
- **Tool_LossAssess**：损失评估辅助工具
- **ReadRecord**：地震记录读取工具