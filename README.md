# MDOFModel

Generate Multi-Degree-Of-Freedom (MDOF) structures based on basic building information (such as floor area, number of stories, etc.), and perform dynamic analysis and economic loss assessment.

[中文文档 (Chinese Documentation)](https://github.com/youtian95/MDOFModel/blob/master/README_CN.md)

## Project Introduction

MDOFModel is a Python library for Multi-Degree-Of-Freedom (MDOF) model analysis in structural engineering, primarily for seismic engineering analysis. This tool can generate lumped mass finite element models from basic building information and perform dynamic analysis, pushover analysis, loss assessment, and Incremental Dynamic Analysis (IDA).

## Features

- **Structure Model Generation**: Generate MDOF structural models based on basic building parameters (such as number of stories, floor area, structural type)
- **Seismic Dynamic Analysis**: Perform linear and nonlinear time history analysis
- **Pushover Analysis**: Implement structural static pushover analysis
- **Incremental Dynamic Analysis (IDA)**: Execute IDA analysis using FEMA P-695 far-field earthquake records
- **Loss Assessment**: Conduct seismic loss assessment based on Hazus methodology
- **OpenSees Integration**: Seamless integration with OpenSees

## Installation Guide

Install this library using pip:

```bash
pip install MDOFModel
```

### Dependencies

- Python >= 3.12
- numpy
- pandas
- matplotlib
- openseespy
- openpyxl
- eqsig

## Usage Examples

### Example 1: Dynamic Analysis

```python
from MDOFModel import MDOF_CN as mcn
from MDOFModel import MDOFOpenSees as mops

# Create a 3-story structure model
NumofStories = 3
bld = mcn.MDOF_CN(NumofStories, 1000, 'S2', City='Shijiazhuang',longitude=114.52,latitude=38.05)
bld.OutputStructuralParameters('structural parameters')

# Perform dynamic analysis
fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)
fe.DynamicAnalysis('H-E12140', 3.0, True)

# Plot story drift time history
fe.PlotForceDriftHistory(1)
```

### Example 2: Incremental Dynamic Analysis (IDA)

```python
from MDOFModel import IDA
from MDOFModel import MDOF_LU as mlu
from MDOFModel import MDOFOpenSees as mops
import numpy as np

# Create structural model
NumofStories = 3
bld = mlu.MDOF_LU(NumofStories, 3600, 'S2')
bld.set_DesignLevel('pre-code')

# Set up OpenSees model
fe = mops.MDOFOpenSees(NumofStories, [bld.mass]*bld.N, [bld.K0]*bld.N, bld.DampingRatio,
    bld.HystereticCurveType, bld.Vyi, bld.betai, bld.etai, bld.DeltaCi, bld.tao)

# Perform IDA analysis
IM_list = np.linspace(0.1, 2.0, 10).tolist()
IDA_obj = IDA.IDA(fe)
IDA_result = IDA_obj.Analyze(IM_list, EQRecordFile_list, bld.T1)

# Save and plot results
IDA_result.to_csv('IDA_results.csv')
IDA.IDA.plot_IDA_results(IDA_result, Stat=True, FigName='IDA.jpg')
```

## Main Modules Description

- **MDOF_CN**: Multi-degree-of-freedom model generation based on Chinese codes
- **MDOF_LU**: General multi-degree-of-freedom model generation
- **MDOFOpenSees**: OpenSees interface for modeling and analysis
- **IDA**: Incremental Dynamic Analysis
- **BldLossAssessment**: Building loss assessment
- **Tool_IDA**: IDA analysis auxiliary tools
- **Tool_LossAssess**: Loss assessment auxiliary tools
- **ReadRecord**: Earthquake record reading tool