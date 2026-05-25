# 更新日志

## [0.8.0] - 2026-05-25

- [x] 不需要通过积分计算地面速度，MaxAbsVel本身就可以记录
- [x] 修复大BUG：GeneralModelWrapper.py 中速度和加速度记录的第0个都应该是地面，所以速度和加速度的数量都应该比Drift多1个。Pelicun的损失评估中，也有相应需要修改的地方。删除compute_pgv函数，直接记录MaxAbsVel。
- [x] 删除IDA_3D.py和IDA.py。
- [x] 将`Examples\Example_MRF_Model.py`改为一个非常简单的糖葫芦串模型。

## [0.7.0] - 2026-05-23

- [x] 合并 `IDA_3D.py` 和 `IDA.py` 模块，为`IDA_2D.py`，因为都是针对平面结构的分析，区别在于是否记录双向EDP。
- [x] 同步修改原来的例子。
- [x] IDA_2D 模块允许IDA分析随时中断和恢复。

## [0.6.1] - 2026-05-21

- [x] 修改例子7，改为文件内定义接口函数。
- [x] 将IDA进度的主进度条显示改为总数为地震波数量乘以Sa数量，而不是地震波数量。每分析完一个Sa就更新一次主进度条。

## [0.6.0] - 2026-05-20

- [x] Pelicun损失评估允许增加定义新的构件。但是这样EDP类型必须在IDA分析结果中存在。
- [x] 构件EDP类型不存在时，需要在IDA分析结果中增加相应的EDP类型。通过 `IDA_f`/`IDA3DAnalysis.Analyze` 的 `ExtraEDP={'EDP类型名': 'GeneralModelWrapper属性名'}` 参数，将 OpenSees 模型的自定义结构响应量（如柱应变、节点转角等）写入 IDA 结果 CSV，并自动流入 `demand.csv`，供自定义构件使用。
- [x] 根据例子6增加相应的例子（例子7），展示如何定义新的构件和EDP类型，并进行损失评估。

## [0.5.1] - 2026-05-15

- [x] 例子3推覆改为滞回

## [0.5.0] - 2026-05-07

- [x] 增加`IDA_3D.py`模块，对 FEMA P-695 双向地震动记录对同时分析，IM 取几何均值 Sa，记录 X/Y 双向 EDP（MaxDrift_X/Y、MaxAbsAccel_X/Y 等）。对平面结构：将同一结构在两个方向激励下独立分析。
- [x] 将例子4、5、6均改为使用`IDA_3D.py`模块进行分析：例子4直接改用 IDA3DAnalysis；例子5使用 IDA3D_to_2d_envelope 取包络后接 Hazus 方法；例子6通过 IdaCsv_3D 参数将 X/Y 双向 EDP 传入 Pelicun，方向1=X，方向2=Y。
- [x] `CollapseAnalysis` 自动识别 3D IDA CSV（含 MaxDrift_X/Y 列），倒塌判定取两方向最大漂移。
- [x] `PelicunLossAssessment.LossAssessment` 增加 `IdaCsv_3D` 参数，`_build_demand_csv` 支持传入独立的 Y 方向 EDP。
- [x] `LossAssessment` 自动识别 3D CSV 格式，并且IDA输入仅支持csv文件或者DataFrame。
- [x] IDA中计算谱加速度时使用临时文件存储中间结果。

## [0.4.0] - 2026-05-07

- [x] `IDA`模块中，类增加专门的输出为csv文件的函数，输出方式跟`Tool_IDA.py`中输出csv文件的方式一样。
- [x] 增加`Collapse`模块，IDA分析输出CSV文件结果后，应该调用`Collapse`模块。`Collapse`模块包含`filter_collapse`（剔除倒塌记录）和`fit_collapse_fragility`（MLE拟合倒塌易损性中值和对数标准差）。损失评估流程中，IDA分析之后调用Collapse模块得到倒塌概率和筛选后的IDA，然后进行损失评估。
- [x] `LossAssessment`返回的结果中增加倒塌概率和不可修复概率。

## [0.3.6] - 2026-05-06

- [x] 修改`MDOFModel\loss\PelicunLossAssessment.py`的接口，使其能够直接使用IDA的csv文件。
- [x] 删除例子7。
- [x] 重新跑了例子4的IDA。

## [0.3.5] - 2026-05-06

- [x] 推覆分析绘图纵坐标改为总剪力除以重力，方便与Sa(g)对比
- [x] IDA绘图同时展示单独record的结果，用灰色细线表示
- [x] IDA增加直接根据csv结果文件绘图函数

## [0.3.4] - 2026-05-06

- [x] 测试例子6中倒塌概率
- [x] 例子6没有设置倒塌的维修时间

## [0.3.3] - 2026-05-06

- [x] `MDOFModel\models\GeneralModelWrapper.py`中的单位直接写死规定

## [0.3.2] - 2026-05-06

- [x] 增加Pelicun输出的结果内容

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
