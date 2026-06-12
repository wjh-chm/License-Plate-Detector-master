# License Plate Detector 微调实验总结

## 1. 微调工作的起点
本轮微调是基于项目原始车牌检测权重 `weights/best.pt` 开始的，目标是让模型适应我们自己的数据集，并尽量提高车牌检测率，尤其是提高小目标车牌的召回率。

本项目不是普通的 5 列 YOLO 标签训练，而是使用 13 列标签：`class cx cy w h x1 y1 x2 y2 x3 y3 x4 y4`。
其中前 5 列是普通检测框，后 8 列是 4 个角点坐标。由于我们的原始标注本质上是矩形框，所以这批 13 列标签中的角点是由 bbox 四角展开得到的“伪角点”，它的主要作用是兼容当前项目的训练链路，而不是做真正的透视几何学习。

## 2. 使用了哪些数据集场景
### 2.1 原始微调数据集
本轮微调使用的外部数据集是：
`D:/project/dataloader/exports/fusion_round1_quad_dataset`

训练集场景一共 8 个：
- `car_far`
- `car_far1`
- `car_far_long`
- `GF1402-2`
- `GF1402-3`
- `side_car`
- `side_car_long`
- `side_car_short`

验证集场景一共 2 个：
- `1867C-loom`
- `1867C-looming2`

原始整图数据规模为：
- 训练集：359 张
- 验证集：120 张

其中验证集中正样本数量较少，只有 40 张带车牌目标的图像，其余为无目标样本。因此验证集对误检和漏检都比较敏感。

### 2.2 切图训练数据集
为了应对小目标问题，后续我们从原始数据集生成了切图版数据集：
`D:/project/dataloader/exports/fusion_round1_quad_dataset_tiles_192_o64`

切图策略为：
- tile 尺寸：`192 x 192`
- overlap：`64 x 64`

切图后数据规模变为：
- 训练集：2154 个 tiles
- 验证集：720 个 tiles

切图训练集统计：
- 正样本 tiles：1472
- 空标签 tiles：682

切图验证集统计：
- 正样本 tiles：234
- 空标签 tiles：486

需要注意的是，切图数据集并没有改变训练/验证场景划分，只是把原有整图按固定窗口切成了更小的图块。

## 3. 训练后如何判断效果
本轮实验主要使用以下指标判断训练效果：

### 3.1 Precision
Precision 表示“模型报出来的框里，有多少是真正正确的”。
如果 Precision 低，说明误检多。

### 3.2 Recall
Recall 表示“真实存在的车牌里，模型找到了多少”。
如果 Recall 低，说明漏检多。

### 3.3 mAP@0.5
这是目标检测里最常用的整体指标之一。它要求预测框与真实框的 IoU 大于等于 0.5，数值越高说明整体检测能力越好。

### 3.4 mAP@0.5:0.95
这是更严格的综合指标，会在多个 IoU 阈值下统计平均表现，通常比 mAP@0.5 更难看上去好。

### 3.5 IoU
IoU 表示预测框和真实框的重叠程度。
- IoU 越大，说明预测框位置越准。
- 一般 `IoU >= 0.5` 会认为一次检测命中是有效的。

### 3.6 额外使用的“整图切图评估”
在切图方案阶段，我们除了看训练日志里的 tile 级指标，还额外做了“整图切图推理评估”。
也就是说：
- 输入仍然是原始整图
- 推理时按小块切开检测
- 再把检测结果映射回整图并做全局 NMS

这样评估更接近真实部署时的效果，因此在切图阶段，这组整图评估结果比单纯看 tile 验证更有参考意义。

## 4. 第一阶段：整图直接微调
### 4.1 第一版整图微调
训练命令入口是 `train.py`，使用数据配置和超参数配置接入外部数据集后，首先做了一轮整图微调。

这一步的主要工作包括：
- 新增数据配置文件，直接指向外部数据集路径
- 新增带 `landmark` 字段的微调超参数文件
- 修复训练、验证和环境兼容问题，例如：
  - `numpy` 兼容问题
  - `torch` 2.x 类型问题
  - `TensorBoard` 缺失时的容错
  - 13 列标签评估逻辑的 bug
  - `best.pt` 恢复架构时默认 `cfg` 冲突的问题

修复评估链路后，第一版整图微调在验证集上的实际结果为：
- Precision：`0.017`
- Recall：`0.125`
- mAP@0.5：`0.00715`
- mAP@0.5:0.95：`0.00239`

### 4.2 第一版结果说明
这一步说明模型已经能够在新数据上开始学习，但效果很差，主要问题是：
- 召回率偏低，漏检很多
- 精度非常低，误检较多
- mAP 很低，整体检测效果不理想

我们在可视化排查中看到，模型并不是完全没有学到东西，但它对新场景中的小车牌目标不稳定，经常出现“有候选框但置信度排序不对”或者“定位不稳”的情况。

## 5. 第二阶段：提高整图分辨率并减弱角点干扰
### 5.1 为什么提出这个方案
第一版整图微调之后，我们观察到一个核心问题：
车牌在整图中太小，轻量 backbone 在整图上直接看这种小目标非常吃力。

同时，当前 13 列标签中的角点并不是真实透视角点，而是 bbox 四角的伪角点。因此 landmark 分支对“提升检测框质量”的帮助有限，反而可能分走一部分训练能力。

所以第二阶段提出了两个改动：
- 把训练分辨率从 `800` 提高到 `1024`
- 把 `landmark` loss 权重从 `0.005` 降到 `0.001`

### 5.2 第二版整图微调结果
第二版整图微调后，验证集结果为：
- Precision：`0.153`
- Recall：`0.100`
- mAP@0.5：`0.0603`
- mAP@0.5:0.95：`0.0277`

### 5.3 这一阶段的变化
相比第一版整图微调：
- Precision 明显提高
- mAP 明显提高
- 说明框质量和整体检测质量改善了

但问题是：
- Recall 没有提高，反而从 `0.125` 降到了 `0.100`

这意味着模型变得“更谨慎、更干净”了，但“尽量别漏检”这个目标并没有真正解决。

## 6. 第三阶段：提出切图训练与切图推理方案
### 6.1 为什么提出切图方案
第二阶段之后，我们基本确认了主要瓶颈不是单纯的学习率或普通增强，而是“小目标在整图里太小”。

因此提出切图方案，核心思路是：
- 训练时不再直接让模型看整张大图
- 而是把大图切成小块，让车牌在每个 tile 中占据更大的相对面积
- 推理时也用同样方式先切图，再把结果合并回整图

这样做的理由是：
- 对小目标更友好
- 能明显提高召回率
- 比单纯继续堆高整图分辨率更直接有效

### 6.2 第一版切图模型
第一版切图模型训练后，我们用整图切图评估来测部署效果。
在 `conf=0.05, merge_iou=0.3` 条件下，结果为：
- TP：`25`
- FP：`217`
- FN：`15`
- Precision：`0.103`
- Recall：`0.625`

### 6.3 第一版切图结果说明
这一步的意义非常大：
- Recall 从整图方案的 `0.100~0.125`，一下子提升到了 `0.625`
- 说明切图方案对小目标车牌检测非常有效

但同时也出现了新问题：
- FP 高达 `217`
- 误检极其明显

也就是说，第一版切图模型解决了“看得见”的问题，但还没有解决“看得准”的问题。

## 7. 第四阶段：第二轮切图调参，目标是在保持高召回的同时压低误检
### 7.1 为什么提出第二轮切图调参
第一版切图模型的方向是正确的，但误检太多，不适合直接用。

因此第二轮切图调参主要目标是：
- 尽量保住切图带来的高召回
- 同时把误检压下来

为此，我们做了如下改动：
- 从第一版切图模型继续微调，而不是从头开始
- 使用更低学习率
- 保持较低的 `landmark` 权重
- 关闭 `mosaic`
- 使用更“干净”的增强策略，减少背景过拟合带来的噪声误报

### 7.2 第二轮切图结果
第二轮切图模型在整图切图评估下的代表性结果如下。

在 `conf=0.05` 下：
- TP：`22`
- FP：`20`
- FN：`18`
- Precision：`0.524`
- Recall：`0.550`

在 `conf=0.07` 下：
- Precision：`0.692`
- Recall：`0.450`

### 7.3 第二轮切图的变化
和第一版切图模型相比：
- FP 从 `217` 大幅下降到 `20`
- Recall 从 `0.625` 下降到 `0.550`

这一步可以理解为：
- 模型从“高召回但非常嘈杂”
- 变成了“召回仍然较高，同时误检已经明显可控”

这是一个非常关键的阶段性改进，因为它首次把“能找到”与“不要乱报”平衡到了一个可用区间。

## 8. 第五阶段：第三轮切图调参，引入 hard negative 微调
### 8.1 为什么提出 hard negative 方案
虽然第二轮切图模型已经把误检压下来了，但验证中仍然能找到一些稳定的错误背景区域。

因此我们提出 hard negative 微调方案：
- 不使用验证集反哺训练，避免数据泄漏
- 只从训练集中的空标签 tiles 里挖掘“模型最容易误报的背景块”
- 将这些 hard negatives 额外重复加入训练列表
- 用更短、更稳、更低学习率的方式再做一轮校正

### 8.2 hard negative 挖掘结果
在训练集的 682 个空标签 tile 中：
- 有 280 个会被第二轮切图模型误报
- 最终选取了 160 个最典型的 hard negatives
- 每个重复 3 次，混入第三轮训练集

### 8.3 第三轮 hard negative 训练的工程调整
由于第三轮 hard-negative 训练会带来更大的显存压力，我们做了额外处理：
- 降低 batch size
- 将短程 hard-negative 微调放到 `640` 分辨率执行
- 修复 `train.py` 中短 epoch 场景下不写 `results.txt`、不保存权重的问题，使短程实验也能稳定产出模型

### 8.4 第三轮结果
第三轮 hard-negative 模型在整图切图评估下：

在 `conf=0.04` 下：
- TP：`20`
- FP：`12`
- FN：`20`
- Precision：`0.625`
- Recall：`0.500`

在 `conf=0.05` 下：
- TP：`17`
- FP：`4`
- FN：`23`
- Precision：`0.810`
- Recall：`0.425`

### 8.5 第三轮结果说明
与第二轮切图模型相比：
- 误检继续下降
- Precision 进一步提升
- 但 Recall 继续下降

这说明 third-stage hard negative 微调确实起到了“去假阳性”的作用，但代价是模型变得更保守，一些边缘样本开始漏掉。

## 9. 当前各阶段的总体结论
### 9.1 整图直接微调
优点：
- 流程简单
- mAP 有提升

缺点：
- 对小目标车牌不友好
- Recall 太低

结论：
- 不适合作为最终方案

### 9.2 切图第一版
优点：
- Recall 大幅提升
- 明确证明“小目标问题”是主要瓶颈

缺点：
- 误检太多

结论：
- 方向完全正确，但还不能直接用

### 9.3 切图第二版
优点：
- 在保持较高召回的同时，把误检压下来了
- 综合平衡最好

缺点：
- 仍然有一些误检和少量漏检

结论：
- 目前最均衡、最推荐作为主力版本

### 9.4 切图第三版 hard negative
优点：
- 进一步降低误检
- Precision 非常高

缺点：
- Recall 比第二版再下降一截

结论：
- 更适合“宁可少报也不要乱报”的场景
- 如果当前业务更在意检测率，第三版不一定比第二版更适合作为默认主模型

## 10. 目前推荐的模型使用建议
如果当前业务目标是“尽量提高检测率，同时把误检控制在可接受范围内”，建议优先使用：
- 第二轮切图模型：`runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt`

推荐推理参数：
- `conf-thres=0.05`
- `merge-iou-thres=0.3`

如果业务目标更偏向“误检尽量少”，可以尝试：
- 第三轮 hard-negative 模型：`runs/train/fusion_round1_quad_tiles_ft_v3_hardneg_640bs2_e3/weights/best.pt`

推荐推理参数：
- 更均衡：`conf-thres=0.04`
- 更严格：`conf-thres=0.05`

## 11. 可视化分析补充
为了直观比较第二轮和第三轮模型的差异，我们还生成了 `GT / V2 / V3` 三栏并排图，用于观察：
- 哪些误检被第三轮清掉了
- 哪些图第三轮更干净
- 哪些图第三轮开始漏检

这些可视化结果说明：
- 第三轮 hard negative 微调确实能清掉一部分假框
- 但它也会让模型在一些边缘样本上变得更保守
- 这与我们从 Precision 和 Recall 上看到的趋势一致

## 12. 本轮工作整体结论
从模型微调开始到现在，整体工作路径可以总结为：
- 先把外部数据集接入当前项目，并修好训练/验证链路
- 先做整图微调，确认原方案在小目标数据上效果不够
- 通过提升分辨率和减弱角点监督，改善框质量
- 进一步确认小目标才是核心瓶颈后，转向切图训练与切图推理
- 利用切图方案大幅提高召回率
- 再通过 cleaner augmentation 和 hard negative，逐步把误检压下来

最终得到的经验是：
- 对这套数据，最关键的改进不是普通学习率微调，而是小目标处理策略
- 切图训练和切图推理是本轮效果提升的核心
- 第二轮切图模型提供了当前最好的“召回与误检平衡”
- 第三轮 hard negative 模型则适合追求更低误检的场景

## 13. 本轮修改过的代码文件清单及作用
### 13.1 训练与验证主流程
- `train.py`
  - 当前项目的训练入口脚本。
  - 本轮主要修改了：
    - 兼容没有 TensorBoard 的环境
    - 只在 `--resume` 时恢复 optimizer，避免普通微调错误恢复旧训练状态
    - 默认 `cfg` 为空，保证从 `best.pt` 微调时能够正确恢复 checkpoint 自带架构
    - 修复短程训练时最后一轮不评估、不写 `results.txt`、不保存权重的问题

- `test.py`
  - 当前项目的验证入口脚本。
  - 本轮修复了 13 列车牌标签评估时的逻辑问题，避免把角点列错误当成类别列参与评估。

### 13.2 数据加载与底层兼容
- `utils/face_datasets.py`
  - 当前项目车牌数据集的核心 dataloader。
  - 本轮用于接入外部 13 列标签数据，并修复 `numpy` 兼容问题。
  - 同时确认其支持通过 `txt` 文件读入训练图像路径，这一点被后续 hard-negative 混合训练列表复用。

- `utils/datasets.py`
  - 通用图像/推理相关数据处理工具。
  - 本轮主要修复了 `numpy` 兼容问题。

- `utils/general.py`
  - 包含 NMS、通用工具函数等。
  - 本轮主要修复了 `numpy` 兼容问题，并参与切图推理时的全局 NMS 流程。

- `utils/loss.py`
  - 当前项目的损失函数实现。
  - 本轮主要修复了 PyTorch 2.x 下目标构建与 loss 计算中的类型兼容问题。

### 13.3 切图推理与评估
- `detect_plate_tiled.py`
  - 新增的切图推理脚本。
  - 作用是将整图切成重叠 tiles，逐块检测，再把结果映射回原图并做全局 NMS。
  - 本轮还加入了省内存处理，并将默认 `merge-iou-thres` 改为推荐值 `0.3`。

- `evaluate_tiled_detector.py`
  - 新增的整图切图评估脚本。
  - 用于在原始整图验证集上模拟实际部署方式，计算切图方案的 `precision / recall / tp / fp / fn`。
  - 这是后续比较第二轮与第三轮切图模型时最核心的评估工具之一。

### 13.4 数据生成与 hard-negative 挖掘
- `tools/generate_tiled_plate_dataset.py`
  - 新增的切图数据集生成脚本。
  - 作用是从原始整图数据集自动生成 tile 数据集、tile 标签、`train.txt`、`val.txt` 和 `data.yaml`。

- `tools/mine_hard_negative_tiles.py`
  - 新增的 hard-negative 挖掘脚本。
  - 作用是扫描训练集中的空标签 tiles，找出会被当前模型误报的背景块，并生成：
    - hard negative 专用列表
    - 混合训练列表
  - 这是第三轮 hard-negative 微调的核心工具。

### 13.5 微调配置文件
- `data/fusion_round1_quad.yaml`
  - 第一阶段整图微调的数据配置文件。
  - 指向原始外部整图数据集的 `images/train` 和 `images/val`。

- `data/hyp.finetune_landmark.yaml`
  - 第一阶段整图微调使用的超参数文件。
  - 在原有微调超参数基础上补足 `landmark` 字段，保证当前项目训练链路可用。

- `data/hyp.finetune_landmark_v2.yaml`
  - 第二阶段整图高分辨率微调使用的超参数文件。
  - 将 landmark 权重从 `0.005` 降到 `0.001`，主要目的是减少伪角点监督对检测分支的干扰。

- `data/fusion_round1_quad_tiled_192o64.yaml`
  - 第一、第二轮切图训练使用的数据配置文件。
  - 指向切图后数据集的 `images/train` 和 `images/val`。

- `data/hyp.finetune_landmark_tile_v2.yaml`
  - 第二轮切图调参使用的超参数文件。
  - 采用更低学习率、更干净增强和关闭 `mosaic` 的策略，以降低误检。

- `data/fusion_round1_quad_tiled_192o64_hardneg.yaml`
  - 第三轮 hard-negative 微调的数据配置文件。
  - 训练集不再直接指向目录，而是指向混合后的 `train.txt`，以便重复采样 hard negatives。

- `data/hyp.finetune_landmark_tile_v3_hardneg.yaml`
  - 第三轮 hard-negative 微调超参数文件。
  - 进一步降低学习率，减弱增强，目的是做一次稳定的误检校正。

## 14. 每个阶段生成的文件及用途
### 14.1 第一阶段：整图微调接入与首轮训练
- `data/fusion_round1_quad.yaml`
  - 整图训练/验证数据入口。

- `data/hyp.finetune_landmark.yaml`
  - 首轮整图微调超参数。

- `runs/train/fusion_round1_quad_ft/weights/best.pt`
  - 第一版整图微调得到的最佳权重。

- `runs/train/fusion_round1_quad_ft/weights/last.pt`
  - 第一版整图微调最后一轮权重。

- `runs/train/fusion_round1_quad_ft/results.txt`
  - 第一版整图训练日志，记录每轮训练和验证指标。

- `runs/test/fusion_round1_quad_ft_eval/`
  - 第一版整图微调修正后评估结果目录。
  - 里面包含评估曲线、批次可视化、混淆矩阵等。

- `runs/test/base_best_eval/`
  - 原始 `weights/best.pt` 在同一验证集上的对照评估目录。
  - 用来和微调结果做基线比较。

- `.tmp/val_compare/summary.json`
  - 第一轮整图阶段的人工排查汇总。
  - 用于记录抽样图像中 GT 与预测框的对比情况。

- `.tmp/val_compare/*_compare.jpg`
  - 第一轮整图阶段生成的可视化对比图。

### 14.2 第二阶段：高分辨率整图微调
- `data/hyp.finetune_landmark_v2.yaml`
  - 第二版整图高分辨率微调超参数。

- `runs/train/fusion_round1_quad_ft_v2_1024/weights/best.pt`
  - 第二版整图微调最佳权重。

- `runs/train/fusion_round1_quad_ft_v2_1024/weights/last.pt`
  - 第二版整图微调最后一轮权重。

- `runs/train/fusion_round1_quad_ft_v2_1024/results.txt`
  - 第二版整图训练日志。

- `runs/test/fusion_round1_quad_ft_v2_1024_eval/`
  - 第二版整图微调评估目录。

### 14.3 第三阶段：切图数据集生成与切图第一版
- `D:/project/dataloader/exports/fusion_round1_quad_dataset_tiles_192_o64/`
  - 由整图数据集自动生成的切图数据集根目录。

- `D:/project/dataloader/exports/fusion_round1_quad_dataset_tiles_192_o64/train.txt`
  - 切图训练集图像路径列表。

- `D:/project/dataloader/exports/fusion_round1_quad_dataset_tiles_192_o64/val.txt`
  - 切图验证集图像路径列表。

- `D:/project/dataloader/exports/fusion_round1_quad_dataset_tiles_192_o64/data.yaml`
  - 切图数据集自带的数据配置文件。

- `D:/project/dataloader/exports/fusion_round1_quad_dataset_tiles_192_o64/tile_export_summary.json`
  - 记录切图数量、正负样本数量、切图参数等统计信息。

- `data/fusion_round1_quad_tiled_192o64.yaml`
  - 仓库内部正式使用的切图训练配置文件。

- `runs/train/fusion_round1_quad_tiles_ft_v1/weights/best.pt`
  - 第一版切图训练得到的最佳权重。

- `runs/train/fusion_round1_quad_tiles_ft_v1/weights/last.pt`
  - 第一版切图训练最后一轮权重。

- `runs/train/fusion_round1_quad_tiles_ft_v1/results.txt`
  - 第一版切图训练日志。

- `.tmp/tiled_threshold_sweep_v1.json`
  - 第一版切图模型阈值扫描结果。
  - 用于分析不同 `conf-thres` 下 Precision / Recall 如何变化。

- `.tmp/tiled_eval_pretrain_v2.json`
  - 第二版整图模型在“整图切图推理模式”下的评估结果。
  - 用来证明仅靠切图推理但不切图训练，提升有限。

- `.tmp/tiled_eval_tile_model_v1.json`
  - 第一版切图模型在整图切图评估下的结果汇总。

### 14.4 第四阶段：第二轮切图调参
- `data/hyp.finetune_landmark_tile_v2.yaml`
  - 第二轮切图调参超参数。

- `runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt`
  - 第二轮切图模型最佳权重。

- `runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/last.pt`
  - 第二轮切图模型最后一轮权重。

- `runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/results.txt`
  - 第二轮切图训练日志。

- `.tmp/tiled_eval_tile_model_v2_conf005.json`
  - 第二轮切图模型在 `conf=0.05` 下的整图切图评估结果。

- `.tmp/tiled_eval_tile_model_v2_conf006.json`
  - 第二轮切图模型在 `conf=0.06` 下的整图切图评估结果。

- `.tmp/tiled_eval_tile_model_v2_conf007.json`
  - 第二轮切图模型在 `conf=0.07` 下的整图切图评估结果。

- `.tmp/tiled_eval_tile_model_v2_conf008.json`
  - 第二轮切图模型在 `conf=0.08` 下的整图切图评估结果。

这些文件共同用于寻找第二轮切图模型最合适的部署阈值区间。

### 14.5 第五阶段：第三轮 hard-negative 微调
- `data/fusion_round1_quad_tiled_192o64_hardneg_only.txt`
  - 挖掘出来的 hard-negative 图像列表。
  - 里面只包含训练集中最容易被误报的空标签 tiles。

- `data/fusion_round1_quad_tiled_192o64_hardneg_train.txt`
  - 第三轮训练实际使用的混合训练列表。
  - 它包含：
    - 原始全部切图训练样本
    - 额外重复采样的 hard-negative 样本

- `data/fusion_round1_quad_tiled_192o64_hardneg.yaml`
  - 第三轮 hard-negative 微调数据入口。

- `data/hyp.finetune_landmark_tile_v3_hardneg.yaml`
  - 第三轮 hard-negative 微调超参数文件。

- `.tmp/hardneg_mining_tile_v2.json`
  - hard-negative 挖掘统计结果。
  - 记录扫描了多少空标签 tiles、筛出了多少 hard negatives，以及 top candidate 是哪些图块。

- `runs/train/fusion_round1_quad_tiles_ft_v3_hardneg_640bs2_e3/weights/best.pt`
  - 第三轮 hard-negative 微调最佳权重。

- `runs/train/fusion_round1_quad_tiles_ft_v3_hardneg_640bs2_e3/weights/last.pt`
  - 第三轮 hard-negative 微调最后一轮权重。

- `runs/train/fusion_round1_quad_tiles_ft_v3_hardneg_640bs2_e3/results.txt`
  - 第三轮 hard-negative 微调训练日志。

- `.tmp/tiled_eval_tile_model_v3_conf004.json`
  - 第三轮模型在 `conf=0.04` 下的整图切图评估结果。

- `.tmp/tiled_eval_tile_model_v3_conf005.json`
  - 第三轮模型在 `conf=0.05` 下的整图切图评估结果。

- `.tmp/tiled_eval_tile_model_v3_conf006.json`
  - 第三轮模型在 `conf=0.06` 下的整图切图评估结果。

### 14.6 第六阶段：第二轮与第三轮并排可视化分析
- `.tmp/v2_v3_compare_panels/summary.json`
  - 记录本轮对比中选出的代表性图像及输出面板位置。

- `.tmp/v2_v3_compare_panels/*_panel.jpg`
  - 第二轮模型、第三轮模型与 GT 的三栏并排可视化图。
  - 用于直观观察：
    - 哪些误检被第三轮清掉了
    - 哪些图第三轮更干净
    - 哪些图第三轮产生了漏检或回归

## 15. 当前模型本身的架构说明
### 15.1 本轮微调所使用的实际模型底座
本轮所有微调实验虽然训练策略不断变化，但模型主干架构本身没有更换。
实际加载的基础结构对应的是：
- `models/yolov5n.yaml`

也就是说，本轮整图微调、切图微调、hard-negative 微调，底层使用的都是同一套轻量化 YOLOv5 变体，只是权重不断更新。

这套模型不是标准的 YOLOv5s 主干，而是一个更轻量的版本：
- 前端使用 `StemBlock`
- Backbone 主体使用多层 `ShuffleV2Block`
- Neck 使用 `Conv + Upsample + Concat + C3` 结构
- 检测头使用 3 个尺度的 `Detect`

从结构上看，它可以概括为：
- `StemBlock + ShuffleNetV2 风格 Backbone + YOLOv5/PAN 风格 Neck + 3尺度 Detect Head`

### 15.2 整体结构分为哪几部分
整个模型可以分成 4 个主要部分：
1. 输入 Stem
2. Backbone 主干特征提取
3. Neck 多尺度特征融合
4. Detect Head 检测头

除此之外，这个项目的检测头不是普通 bbox 检测头，而是：
- `bbox + objectness + class + 4个角点`

也就是说，它不仅回归车牌框，还会回归四个角点位置。

### 15.3 输入 Stem：StemBlock
模型最前面不是 YOLOv5 早期常见的 `Focus`，而是 `StemBlock`。

它的作用是：
- 在输入阶段快速下采样
- 同时尽量保留局部细节
- 通过卷积支路和池化支路并联，再拼接融合，得到更稳定的初始特征

可以把它理解成：
- 用一个更适合轻量网络的“特征起步模块”来替代普通单层卷积输入

对你的任务来说，`StemBlock` 的意义在于：
- 车牌本身是小目标
- 输入阶段如果下采样太粗暴，很容易把小目标特征直接冲淡
- `StemBlock` 相比简单卷积更有利于在早期保留有效结构信息

### 15.4 Backbone：ShuffleV2Block 主干
Backbone 主体由多层 `ShuffleV2Block` 构成，整体是明显的 `ShuffleNetV2` 风格。

在 `models/yolov5n.yaml` 中，它的骨干层次大致是：
- `StemBlock`
- `ShuffleV2Block` 下采样到 P3
- 多层 `ShuffleV2Block` 提取中层特征
- `ShuffleV2Block` 下采样到 P4
- 更多 `ShuffleV2Block` 提取更深层特征
- `ShuffleV2Block` 下采样到 P5

`ShuffleV2Block` 的核心作用是：
- 用更低的计算成本提取特征
- 通过分支结构和 `channel shuffle` 保持信息流动
- 在轻量模型里尽可能提高速度和效率

对你的项目来说，这个主干的优点是：
- 轻量
- 推理快
- 对部署友好

缺点也很明显：
- 对特别小、特别难的目标，能力会比更重的 backbone 弱一些

这也是为什么我们后面必须引入切图方案，因为光靠轻量 backbone 直接看整图，很难把很小的车牌稳定检出来。

### 15.5 Neck：多尺度特征融合模块
Backbone 提取完特征后，模型并不是直接检测，而是进入 Neck 做多尺度融合。

Neck 的主要结构是：
- `Conv`
- `Upsample`
- `Concat`
- `C3`

这部分本质上是 YOLOv5 常见的 PAN/FPN 风格结构，作用是：
- 把深层语义特征往上采样
- 和浅层高分辨率特征拼接
- 形成适合不同大小目标的多尺度特征图

在当前模型里，Neck 最后形成 3 个输出尺度：
- `P3/8`
- `P4/16`
- `P5/32`

这 3 个尺度分别更适合：
- `P3`：较小目标
- `P4`：中等目标
- `P5`：较大目标

对车牌任务来说：
- 小车牌主要依赖 `P3`
- 稍大一些的车牌会由 `P4` 和 `P5` 辅助处理

这也是为什么切图方案会有效。
因为切图之后，原本整图里的极小车牌，在 tile 中会“相对变大”，从而更容易落在 `P3` 这一级被模型有效感知。

### 15.6 C3 模块在 Neck 中的作用
Neck 中的 `C3` 模块是 YOLOv5 常见的特征融合模块。

它的作用可以理解为：
- 在特征拼接之后继续做非线性变换
- 保留一部分捷径路径
- 让融合后的特征既有表达力，又不会太重

在你的模型里，`C3` 主要负责：
- 将 backbone 的不同层特征和上采样后的特征重新整合
- 使多尺度输出更适合检测头读取

### 15.7 Detect Head：三尺度检测头
模型最后的 `Detect` 模块负责真正输出检测结果。

它不是单尺度检测，而是在 3 个尺度同时输出：
- `P3`
- `P4`
- `P5`

每个尺度都对应一组 anchors，因此整个模型能够处理不同尺寸的车牌目标。

当前配置中的 anchors 大致是：
- P3：更小的 anchor
- P4：中等 anchor
- P5：更大的 anchor

这部分的作用是：
- 将多尺度特征转成最终的预测框、置信度和角点结果

### 15.8 这个 Detect Head 与普通 YOLO 的不同点
普通 YOLO 检测头通常输出：
- `x y w h`
- `objectness`
- `class`

而当前项目的 `Detect` 头输出的是：
- `x y w h`
- `objectness`
- `4个角点 = 8个数`
- `class`

也就是总输出维度为：
- `nc + 5 + 8`

在当前单类别车牌任务里，就是：
- `1 + 5 + 8 = 14` 个输出维度

这意味着模型不仅会判断“这里有没有车牌”，还会同时回归：
- 左上角
- 右上角
- 右下角
- 左下角

四个角点位置。

### 15.9 角点分支在当前项目中起什么作用
当前项目的角点分支主要有两个作用：
1. 训练时参与 landmark loss
2. 推理后为 OCR 前的透视矫正提供四个角点

也就是说，检测模型输出的 4 个点，后续可以交给 OCR 前处理脚本去做透视拉正。

但要注意：
- 你当前这批训练数据中的角点不是真实人工标注的透视四角
- 而是由 bbox 四角直接展开得到的伪角点

所以在本轮微调中，角点分支的真实意义更多是：
- 保持与当前项目链路兼容
- 让模型输出格式不变

而不是指望它真正学会高质量透视几何。

这也是为什么我们后面把 `landmark` loss 权重调低，因为它对当前任务“提升检测率”的帮助有限。

### 15.10 当前模型中哪些模块对本轮效果提升最关键
从整个实验过程来看，不同模块在本轮任务中的重要性并不一样。

#### 1. Backbone 的作用
Backbone 负责基础特征提取，是模型“看见目标”的基础。
但因为它本身较轻量，对整图中的极小车牌能力有限，所以仅靠 backbone 本身并不能解决问题。

#### 2. Neck 和多尺度检测头的作用
Neck + Detect Head 对小目标很关键，因为它们提供了 `P3/P4/P5` 三尺度检测能力。
尤其是 `P3`，对小车牌最重要。

#### 3. 切图方案为什么能起作用
切图本质上并没有改变模型结构，而是改变了“目标在输入中的相对大小”。
这使得：
- 原本在整图中太小、很难被 backbone + P3 感知的目标
- 在切图后更容易被同一套结构检测出来

因此本轮效果提升最大的，不是换了某个模块，而是：
- 在保留现有模型结构的前提下，通过切图让现有结构更擅长处理这批小目标数据

#### 4. hard negative 为什么有效
hard-negative 也没有改动主结构。
它的本质是：
- 利用当前模型已经学到的表示能力
- 进一步纠正 objectness / 分类分支在背景上的误判

因此它主要作用在“误检校正”而不是“结构升级”。

### 15.11 对当前模型架构的总体评价
对于你的任务，这套模型架构的优点是：
- 轻量
- 推理速度快
- 已经集成了角点输出，兼容 OCR 链路
- 配合切图后，对小车牌任务有明显可用性

缺点是：
- Backbone 偏轻，对整图极小目标先天不占优
- 如果不做切图，小目标召回较差
- 当前角点监督不是真实透视标注，因此角点分支对几何建模的帮助有限

因此，本轮实验最终得到的结论可以补充为：
- 当前模型结构本身是合理的，尤其适合做轻量化部署
- 真正限制效果的主要不是“模型完全不行”，而是“轻量结构直接看整图小目标太吃亏”
- 所以本轮提升效果最关键的，是围绕这套结构设计了更适合小目标的训练与推理策略，而不是直接换掉整个架构

## 16. 补充实验：在 Neck 的 C3 后加入 ECA 注意力
### 16.1 为什么做这个实验
在确认切图方案是当前主要收益来源之后，又尝试了一个结构级改动：
- 在 Neck 的 4 个 `C3` 模块中加入 ECA channel attention
- 插入位置为 `C3` 内部 `cv3` 之后、模块输出之前
- 目的是让 Neck 融合后的通道特征经过一次轻量通道重标定，观察是否能降低误检或提升暗光场景鲁棒性

### 16.2 具体代码与模型配置
本次新增了：
- `models/layers/eca.py`
  - 新增 `C3ECA` 类
  - 结构为原始 `C3` 主体后接 `EcaModule`
  - forward 流程为：`cv1/cv2 -> concat -> cv3 -> ECA -> output`

- `models/yolov5n_eca.yaml`
  - 基于原始 `models/yolov5n.yaml`
  - 只把 Neck 中 4 个 `C3` 替换为 `C3ECA`
  - Backbone、Detect Head、anchors、类别数均保持不变

- `models/yolo.py`
  - 让模型解析器能够识别 `C3ECA`
  - 保持 `C3ECA` 与 `C3` 一样参与 depth 参数展开

### 16.3 训练方式
ECA 模型不是从头训练，而是从当前推荐的第二轮切图模型继续迁移训练：
- 初始权重：`runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt`
- 新模型配置：`models/yolov5n_eca.yaml`
- 数据配置：`data/fusion_round1_quad_tiled_192o64.yaml`
- 超参数：`data/hyp.finetune_landmark_tile_v2.yaml`
- epoch、batch size、img size 等训练参数保持 V2 不变

训练产物为：
- `runs/train/fusion_round1_quad_tiles_ft_v2_eca/weights/best.pt`
- `runs/train/fusion_round1_quad_tiles_ft_v2_eca/weights/last.pt`

模型构建时，从 V2 权重成功迁移了大部分参数：
- `Transferred 498/504 items`

未迁移的部分主要是新增 ECA 模块参数。

### 16.4 三个子集上的整图切图评估
评估参数保持和 V2 对照一致：
- `conf-thres=0.05`
- `merge-iou-thres=0.3`
- tile 尺寸：`192 x 192`
- overlap：`64 x 64`

#### 1. 亮场景：`side_car_short`
V2 原模型：
- TP：`30`
- FP：`128`
- FN：`0`
- Precision：`0.190`
- Recall：`1.000`

ECA 模型：
- TP：`30`
- FP：`16`
- FN：`0`
- Precision：`0.652`
- Recall：`1.000`

结论：
- ECA 在这个亮场景上明显减少误检
- Recall 没有下降
- 对 `side_car_short` 这类亮场景是有帮助的

#### 2. 已见暗场景：`GF1402-3`
V2 原模型：
- TP：`43`
- FP：`36`
- FN：`12`
- Precision：`0.544`
- Recall：`0.782`

ECA 模型：
- TP：`37`
- FP：`35`
- FN：`18`
- Precision：`0.514`
- Recall：`0.673`

结论：
- ECA 在 `GF1402-3` 上没有带来收益
- FP 基本没有下降
- Recall 从 `0.782` 降到 `0.673`
- 对已见暗场景反而变得更保守

#### 3. 未见暗场景：`1867C-loom + 1867C-looming2`
V2 原模型：
- TP：`22`
- FP：`20`
- FN：`18`
- Precision：`0.524`
- Recall：`0.550`

ECA 模型：
- TP：`8`
- FP：`16`
- FN：`32`
- Precision：`0.333`
- Recall：`0.200`

结论：
- ECA 在原验证集上明显退化
- Recall 从 `0.550` 降到 `0.200`
- Precision 也从 `0.524` 降到 `0.333`
- 对未见暗场景泛化不利

### 16.5 本次 ECA 实验结论
这次 ECA 改动不是完全无效，它在亮场景 `side_car_short` 上确实显著减少了误检。

但从整体目标看，当前项目更关心“亮/暗场景都尽量稳定”，尤其原验证集本身就是未见暗场景。因此这次 ECA 版本不适合作为默认主模型。

当前结论是：
- 如果只看亮场景误检控制，`C3ECA` 有明显帮助
- 如果看暗场景，尤其未见暗场景，`C3ECA` 会明显降低召回
- 当前默认主模型仍建议保持第二轮切图模型：`runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt`
- `runs/train/fusion_round1_quad_tiles_ft_v2_eca/weights/best.pt` 更适合作为结构消融结果保留，而不是替代 V2

## 17. 补充实验：在 Neck 的 C3 后加入 SE 注意力
### 17.1 为什么做这个实验
在 ECA 实验之后，又用同样方式测试了 SE channel attention：
- 在 Neck 的 4 个 `C3` 模块中加入 SE attention
- 插入位置同样为 `C3` 内部 `cv3` 之后、模块输出之前
- 目标是观察 SE 是否能像 ECA 一样降低误检，并进一步判断它对亮场景和暗场景是否更稳定

### 17.2 具体代码与模型配置
本次新增了：
- `models/layers/squeeze_excite.py`
  - 新增 `C3SE` 类
  - 结构为原始 `C3` 主体后接 `SEModule`
  - forward 流程为：`cv1/cv2 -> concat -> cv3 -> SE -> output`

- `models/yolov5n_se.yaml`
  - 基于原始 `models/yolov5n.yaml`
  - 只把 Neck 中 4 个 `C3` 替换为 `C3SE`
  - Backbone、Detect Head、anchors、类别数均保持不变

- `models/yolo.py`
  - 让模型解析器能够识别 `C3SE`
  - 保持 `C3SE` 与 `C3` 一样参与 depth 参数展开

### 17.3 训练方式
SE 模型同样不是从头训练，而是从当前推荐的第二轮切图模型继续迁移训练：
- 初始权重：`runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt`
- 新模型配置：`models/yolov5n_se.yaml`
- 数据配置：`data/fusion_round1_quad_tiled_192o64.yaml`
- 超参数：`data/hyp.finetune_landmark_tile_v2.yaml`
- epoch、batch size、img size 等训练参数保持 V2 不变

训练产物为：
- `runs/train/fusion_round1_quad_tiles_ft_v2_se/weights/best.pt`
- `runs/train/fusion_round1_quad_tiles_ft_v2_se/weights/last.pt`

模型构建时，从 V2 权重成功迁移了大部分参数：
- `Transferred 498/516 items`

未迁移的部分主要是新增 SE 模块参数。

### 17.4 三个子集上的整图切图评估
评估参数保持和 V2、ECA 对照一致：
- `conf-thres=0.05`
- `merge-iou-thres=0.3`
- tile 尺寸：`192 x 192`
- overlap：`64 x 64`

#### 1. 亮场景：`side_car_short`
V2 原模型：
- TP：`30`
- FP：`128`
- FN：`0`
- Precision：`0.190`
- Recall：`1.000`

SE 模型：
- TP：`30`
- FP：`17`
- FN：`0`
- Precision：`0.638`
- Recall：`1.000`

结论：
- SE 在亮场景上明显减少误检
- Recall 没有下降
- 这一点和 ECA 实验趋势一致

#### 2. 已见暗场景：`GF1402-3`
V2 原模型：
- TP：`43`
- FP：`36`
- FN：`12`
- Precision：`0.544`
- Recall：`0.782`

SE 模型：
- TP：`38`
- FP：`27`
- FN：`17`
- Precision：`0.585`
- Recall：`0.691`

结论：
- SE 在 `GF1402-3` 上降低了一部分误检，Precision 从 `0.544` 提升到 `0.585`
- 但 Recall 从 `0.782` 降到 `0.691`
- 说明 SE 对已见暗场景更偏向“保守去误检”，不是纯收益

#### 3. 未见暗场景：`1867C-loom + 1867C-looming2`
V2 原模型：
- TP：`22`
- FP：`20`
- FN：`18`
- Precision：`0.524`
- Recall：`0.550`

SE 模型：
- TP：`5`
- FP：`9`
- FN：`35`
- Precision：`0.357`
- Recall：`0.125`

结论：
- SE 在原验证集上明显退化
- Recall 从 `0.550` 降到 `0.125`
- 虽然 FP 从 `20` 降到 `9`，但漏检代价太大
- 对未见暗场景泛化不利

### 17.5 本次 SE 实验结论
SE 和 ECA 的整体趋势接近：都能明显压低亮场景误检，但都会让模型在暗场景，尤其未见暗场景上变得更保守。

当前结论是：
- 如果只看亮场景误检控制，`C3SE` 有明显帮助
- 在已见暗场景 `GF1402-3` 上，`C3SE` 能小幅提高 Precision，但会牺牲 Recall
- 在未见暗场景 `1867C` 上，`C3SE` 明显不适合，Recall 下降过大
- 当前默认主模型仍建议保持第二轮切图模型：`runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt`
- `runs/train/fusion_round1_quad_tiles_ft_v2_se/weights/best.pt` 更适合作为结构消融结果保留，而不是替代 V2
