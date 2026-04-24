# MDOFModel

根据建筑基本信息（如建筑面积、层数等）生成多自由度(MDOF)结构，并进行动力分析和经济损失评估。

[English Documentation](README.md)

## 项目介绍

MDOFModel 是一个用于结构工程中多自由度（MDOF）模型分析的 Python 库，主要用于地震工程分析。该工具可以通过基本建筑信息自动生成集中质量有限元模型，进行动力时程分析、静力推覆分析、损失评估以及增量动力分析（IDA）。

## 功能特点

- **结构模型生成**：基于基本建筑参数（如层数、建筑面积、结构类型）生成 MDOF 结构模型
- **地震动力分析**：执行线性和非线性时程分析
- **推覆分析 (Pushover)**：实施结构静力推覆分析
- **增量动力分析 (IDA)**：采用 FEMA P-695 远场地震动记录执行 IDA 分析
- **损失评估**：基于 Hazus 方法进行地震经济损失评估

## 安装说明

使用 pip 安装本库：

```bash
pip install MDOFModel
```

## 使用示例

详细的使用示例请参考本代码库中的 `Examples` 目录。我们提供了几个可直接独立运行的脚本形式的例子，用于演示 MDOFModel 各种不同的功能：

- **Example1_ShearBuildingModel**：包含一系列演示简化剪切型（Shear Building）建筑模型的脚本集合：
  - `1_Dynamic.py`：时程动力分析。
  - `2_Pushover.py`：静力推覆分析。
  - `3_LossAssessment.py`：经济损失评估。
  - `4_IDA.py`：增量动力分析 (IDA)。
  - `5_EQSpectra.py`：地震动反应谱处理与绘图。

- **Example2_GeneralModel_Dynamic**：演示如何使用 `GeneralModelWrapper` 包装器对一般 OpenSees 结构模型（例如二维框架结构）进行动力时程分析。

- **Example3_GeneralModel_Pushover**：演示如何使用 `GeneralModelWrapper` 对一般 OpenSees 结构模型进行静力推覆分析并可视化结构。

- **Example4_GeneralModel_IDA**：演示如何使用 `GeneralModelWrapper` 对一般 OpenSees 结构模型执行多线程增量动力分析 (IDA)。

## 主要模块说明

- **MDOF_CN**：基于中国规范的多自由度模型生成
- **MDOF_LU**：通用的多自由度模型生成
- **MDOFOpenSees**：用于建模和分析的 OpenSees 接口
- **IDA**：增量动力分析计算模块
- **BldLossAssessment**：建筑损失评估模块
- **Tool_IDA**：IDA 分析辅助后处理工具
- **Tool_LossAssess**：损失评估辅助工具
- **ReadRecord**：地震动记录读取与解析工具
