# Step 1：候选区域引导切图

## 1. 目标
把当前“固定滑窗切图推理”升级成“候选区域引导切图推理”，先尽量减少无效 tile，再保持或提升 Recall。

这一阶段先只动推理和评估，不先改训练。

## 2. 默认前提
- 默认仓库根目录：`D:\project\license-plate-detector-master`
- 默认主模型：`runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt`
- 默认整图评估脚本：`evaluate_tiled_detector.py`
- 默认切图推理脚本：`detect_plate_tiled.py`
- 默认数据配置：`data/fusion_round1_quad.yaml`
- 默认切图参数：`tile=192`，`overlap=64`，`merge-iou-thres=0.3`

如果你的脚本参数名和下面模板不同，以 `--help` 为准。

## 3. 先改哪些文件
- `detect_plate_tiled.py`
- `evaluate_tiled_detector.py`

建议新增这些能力：
- `--proposal-mode`：`none` / `proposal_only` / `proposal_plus_sparse`
- `--proposal-weights`
- `--proposal-conf`
- `--proposal-expand-ratio`
- `--max-tiles-per-image`
- 评估输出里增加：
  - `avg_tiles_per_image`
  - `max_tiles_per_image`
  - `avg_latency_ms`
  - `lowlight_image_count` 如果你后面顺手做了亮度统计

## 4. 你先要写的命令

### 4.1 先确认脚本参数
```powershell
cd D:\project\license-plate-detector-master
python detect_plate_tiled.py --help
python evaluate_tiled_detector.py --help
```

### 4.2 跑固定滑窗基线
```powershell
cd D:\project\license-plate-detector-master
python evaluate_tiled_detector.py ^
  --weights runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt ^
  --data data/fusion_round1_quad.yaml ^
  --tile-size 192 ^
  --tile-overlap 64 ^
  --conf-thres 0.05 ^
  --merge-iou-thres 0.3 ^
  --proposal-mode none ^
  --save-json .tmp/step1_baseline_fixed_conf005.json
```

### 4.3 跑候选区域引导版本
先用同一个模型做 proposer，后面如果效果一般，再换成整图模型。
```powershell
cd D:\project\license-plate-detector-master
python evaluate_tiled_detector.py ^
  --weights runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt ^
  --data data/fusion_round1_quad.yaml ^
  --tile-size 192 ^
  --tile-overlap 64 ^
  --conf-thres 0.05 ^
  --merge-iou-thres 0.3 ^
  --proposal-mode proposal_only ^
  --proposal-weights runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt ^
  --proposal-conf 0.02 ^
  --proposal-expand-ratio 2.0 ^
  --max-tiles-per-image 16 ^
  --save-json .tmp/step1_proposal_only_conf005.json
```

### 4.4 跑候选区域 + 稀疏保底滑窗
这是我更推荐的版本，因为更不容易漏掉 proposer 没看到的小牌。
```powershell
cd D:\project\license-plate-detector-master
python evaluate_tiled_detector.py ^
  --weights runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt ^
  --data data/fusion_round1_quad.yaml ^
  --tile-size 192 ^
  --tile-overlap 64 ^
  --conf-thres 0.05 ^
  --merge-iou-thres 0.3 ^
  --proposal-mode proposal_plus_sparse ^
  --proposal-weights runs/train/fusion_round1_quad_tiles_ft_v2_cleanaug_bs4/weights/best.pt ^
  --proposal-conf 0.02 ^
  --proposal-expand-ratio 2.0 ^
  --max-tiles-per-image 16 ^
  --save-json .tmp/step1_proposal_plus_sparse_conf005.json
```

### 4.5 最少做一轮参数扫描
重点扫这三个量：
- `proposal-conf`: `0.01 / 0.02 / 0.03`
- `proposal-expand-ratio`: `1.5 / 2.0 / 2.5`
- `max-tiles-per-image`: `8 / 16 / 32`

如果你不想一次扫太多，先固定：
- `proposal-conf=0.02`
- `proposal-expand-ratio=2.0`
- `max-tiles-per-image=16`

## 5. 你要输出什么

### 5.1 必须输出
- `baseline fixed` 的整图评估 JSON
- `proposal_only` 的整图评估 JSON
- `proposal_plus_sparse` 的整图评估 JSON
- 一张对比表，至少包含：
  - `mode`
  - `precision`
  - `recall`
  - `tp`
  - `fp`
  - `fn`
  - `avg_tiles_per_image`
  - `avg_latency_ms`

### 5.2 最好额外输出
- 三个子集分别的结果：
  - 亮场 `side_car_short`
  - 已见暗场 `GF1402-3`
  - 未见暗场 `1867C-loom + 1867C-looming2`
- 5 到 10 张失败案例可视化：
  - proposer 没覆盖到 GT
  - tile 合并后坐标偏掉
  - 全局 NMS 误删真框

## 6. 你要重点注意什么
- proposer 的阈值要低一点，宁可多给候选，也不要先把小牌筛没了。
- proposal 框一定要外扩，不然切图边缘很容易把小牌切坏。
- 保底滑窗不要一开始就删掉，否则你会很难判断 Recall 掉点是 proposer 问题还是结构问题。
- 映射回整图后的全局 NMS 一定要保持一致，不要边改 proposal 边改 NMS。
- 第一步不要顺手改训练超参、注意力模块、数据增强，不然实验会混。

## 7. 这一阶段最重要的指标

按优先级排序：
1. `Recall`
2. 未见暗场子集 Recall
3. `avg_tiles_per_image`
4. `FP`
5. `avg_latency_ms`

我的判断标准：
- 如果 Recall 基本不掉，且 `avg_tiles_per_image` 明显下降，这一步就是成功的。
- 如果 Recall 略涨且 tile 数不增加太多，这一步非常值。
- 如果 tile 数降了，但未见暗场 Recall 掉很多，这一步不能上线。

## 8. 我建议你的收尾结论格式
你最终最好写成三句话：
- 当前最优模式是哪一个。
- 它相比固定滑窗，Recall 变化多少，平均 tile 数变化多少。
- 它在未见暗场上是否还能保持稳定。
