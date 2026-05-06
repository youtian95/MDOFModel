# 更新日志


- [ ] 增加例子7：基于IDA的地震损失评估（使用预先定义好的完整构件量文件，区别于例子6中通过`normqtypact`自动生成非结构构件量的方式）。
- [ ] EDP读取IDA分析结果时应该剔除掉倒塌的地震波用于后续的损失评估。
- [ ] 增加平面结构的IDA_2D分析，会对两个方向的地震波进行分析，记录两个方向的EDP结果，并且进行Pelicun损失评估时考虑两个方向的结果。

## [0.3.1] - 2026-05-06

- [x] 测试例子6。修改pelicun方法中生成默认config文件单位的bug

## [0.3.0] - 2026-05-06

### 修改

- [x] bug：修复`MDOFModel\loss\BldLossAssessment.py`路径的bug
- [x] 注释全部改为中文。
- [x] IDA 分析时增加记录绝对速度
- [x] `MDOFModel\models\GeneralModelWrapper.py`分析时记录的加速度和速度数据改为绝对值，便于后续损失评估使用。
- [x] interp_edp_from_ida 插值应该进行线性插值，而不是最近值。并且移到IDA模块中。
- [x] `MDOFModel\models\GeneralModelWrapper.py` 中 `DynamicAnalysis` 应该直接记录envolop最大值，包括绝对加速度、相对加速度、绝对速度、相对速度，最大层间位移角等，而不是在后处理中计算。只有残余位移需要根据记录的层间位移角历史的最后几秒的结果平均值作为残余位移角，只有这个需要后处理。

### 增加

- [x] 增加`MDOFModel\loss\PelicunLossAssessment.py`模块，基于Pelicun的评估方法进行地震损失评估。增加对`normqtypact`包的依赖，用于生成Pelicun所需的非结构构件输入文件。
- [x] 测试例子5。

## [0.2.0] - 2026-04-24

- [x] 添加更新日志。
- [x] 重构源代码。
- [x] 新增 `GeneralModelWrapper.py` 通用模型包装器，并更新示例。

## [0.1.3] - 2025-12-02
