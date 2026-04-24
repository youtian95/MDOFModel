# MDOFModel

Generate Multi-Degree-Of-Freedom (MDOF) structures based on basic building information (such as floor area, number of stories, etc.), and perform dynamic analysis and economic loss assessment.

[中文文档 (Chinese Documentation)](README_CN.md)

## Project Introduction

MDOFModel is a Python library for Multi-Degree-Of-Freedom (MDOF) model analysis in structural engineering, primarily for seismic engineering analysis. This tool can generate lumped mass finite element models from basic building information and perform dynamic analysis, pushover analysis, loss assessment, and Incremental Dynamic Analysis (IDA).

## Features

- **Structure Model Generation**: Generate MDOF structural models based on basic building parameters (such as number of stories, floor area, structural type)
- **Seismic Dynamic Analysis**: Perform linear and nonlinear time history analysis
- **Pushover Analysis**: Implement structural static pushover analysis
- **Incremental Dynamic Analysis (IDA)**: Execute IDA analysis using FEMA P-695 far-field earthquake records
- **Loss Assessment**: Conduct seismic loss assessment based on Hazus methodology

## Installation Guide

Install this library using pip:

```bash
pip install MDOFModel
```

## Usage Examples

Please refer to the Examples directory in this repository for detailed usage examples. We provide several ready-to-run scripts demonstrating different capabilities of MDOFModel:

- **Example1_ShearBuildingModel**: A collection of examples demonstrating a simplified shear building model for:
  - 1_Dynamic.py: Time history dynamic analysis.
  - 2_Pushover.py: Static pushover analysis.
  - 3_LossAssessment.py: Economic loss assessment.
  - 4_IDA.py: Incremental Dynamic Analysis (IDA).
  - 5_EQSpectra.py: Earthquake spectra processing.

- **Example2_GeneralModel_Dynamic**: Demonstrates how to perform dynamic time-history analysis on a general OpenSees structural model (e.g., 2D frame) using GeneralModelWrapper.

- **Example3_GeneralModel_Pushover**: Demonstrates how to perform static pushover analysis on a general OpenSees structural model using GeneralModelWrapper.

- **Example4_GeneralModel_IDA**: Demonstrates how to execute Incremental Dynamic Analysis (IDA) on a general OpenSees structural model using GeneralModelWrapper.

## Main Modules Description

- **MDOF_CN**: Multi-degree-of-freedom model generation based on Chinese codes
- **MDOF_LU**: General multi-degree-of-freedom model generation
- **MDOFOpenSees**: OpenSees interface for modeling and analysis
- **IDA**: Incremental Dynamic Analysis
- **BldLossAssessment**: Building loss assessment
- **Tool_IDA**: IDA analysis auxiliary tools
- **Tool_LossAssess**: Loss assessment auxiliary tools
- **ReadRecord**: Earthquake record reading tool
