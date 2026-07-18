# QLoRA Fine-tuning Qwen2.5-1.5B — 保险理赔问答

使用 QLoRA 技术在 RTX 4060 8GB 单卡上微调 Qwen2.5-1.5B，实现保险理赔场景的领域适配。

## 项目亮点

| 技术点 | 说明 |
|--------|------|
| **QLoRA** | 4-bit 量化 + LoRA 低秩适配，8GB 显存即可训练 1.5B 模型 |
| **超参调优** | 9 组对照实验，系统调优 r / alpha / lr / epochs |
| **TensorBoard** | 训练监控 loss / learning_rate / grad_norm |
| **对比可视化** | 自动生成多实验对比 PNG 图 |

## 环境

- Python 3.11
- PyTorch 2.x + CUDA
- transformers + peft + bitsandbytes
- 显存：≥ 8GB

```bash
pip install torch transformers peft bitsandbytes datasets tensorboard matplotlib
```

## 项目结构

```
├── train_lora.py          # 基础训练脚本（内置数据）
├── train_lora_learn.py    # 逐行注释学习版（JSONL 数据）
├── tune_lora.py           # 超参调优脚本（自动跑多组实验）
├── eval_lora.py           # 模型评估脚本
├── plot_training.py       # 对比画图脚本
├── check_quantization.py  # 量化检查工具
├── data/
│   └── insurance_qa.jsonl # 训练数据（100 条保险问答）
├── output/                # 训练输出（logs / plots / 模型）
└── models/                # 基座模型（需自行下载）
```

## 快速开始

### 1. 下载基座模型

```bash
# 从 HuggingFace 下载 Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen2.5-1.5b
```

### 2. 修改模型路径

在 `train_lora_learn.py` 中修改 `MODEL_PATH` 为你的本地路径。

### 3. 训练

```bash
python train_lora_learn.py
```

### 4. 监控

```bash
tensorboard --logdir ./output/logs
# 浏览器打开 http://localhost:6006
```

### 5. 生成对比图

```bash
python plot_training.py
# 图片保存在 output/plots/
```

## 调优实验

### 参数矩阵

| 参数 | 搜索范围 | 最优值 |
|------|---------|--------|
| LoRA rank (r) | 4, 8, 16 | **8** |
| LoRA alpha | 16, 32, 64 | **32** |
| Learning rate | 1e-4, 2e-4, 5e-4 | **2e-4** |
| Epochs | 3, 5 | **5** |
| Dropout | 0.05, 0.10 | **0.05** |

### 实验结果（按 Loss 排序）

| 排名 | 实验 | r | alpha | lr | ep | Loss | 时长 |
|:--:|------|--:|--:|--:|--:|--:|--:|
| 🥇 | r08-a32-lr2e4-ep5 | 8 | 32 | 2e-4 | 5 | **1.45** | 247s |
| 🥈 | r16-a64-lr1e4-ep5 | 16 | 64 | 1e-4 | 5 | 1.49 | 270s |
| 🥉 | r08-a32-lr5e4-ep3 | 8 | 32 | 5e-4 | 3 | 1.68 | 146s |
| 4 | r16-a32-lr2e4-ep3 | 16 | 32 | 2e-4 | 3 | 1.78 | 152s |
| 5 | r08-a64-lr2e4-ep3 | 8 | 64 | 2e-4 | 3 | 1.99 | 157s |
| 6 | r08-a32-lr2e4-ep3 | 8 | 32 | 2e-4 | 3 | 2.00 | 140s |
| 7 | r04-a32-lr2e4-ep3 | 4 | 32 | 2e-4 | 3 | 2.23 | 128s |
| 8 | r04-a16-lr2e4-ep3 | 4 | 16 | 2e-4 | 3 | 2.33 | 136s |
| 9 | r08-a32-lr1e4-ep3 | 8 | 32 | 1e-4 | 3 | 2.42 | 138s |

### 调优结论

1. **epochs 是最有效的参数**：5 epochs 比 3 epochs 提升 28%
2. **r=8 性价比最高**：参数量为 r=16 的一半，效果几乎持平
3. **lr=2e-4 是 sweet spot**：太低收敛慢，太高可能过拟合
4. **alpha/r = 4 最优**：alpha=32, r=8 达到最佳配比

## 面试知识点

### QLoRA 原理

```
W_frozen (4-bit)  ← 基座权重，冻结不训练
      +
  ΔW = A·B        ← LoRA 低秩矩阵，r 控制参数量
      ×
  alpha / r        ← 缩放因子

前向: output = W·x + (alpha/r) · (A·B)·x
```

### 关键技术

- **4-bit NormalFloat (NF4)**：信息论最优的 4-bit 数据类型
- **双重量化**：量化 scale 也量化，再省 0.5GB
- **Paged Optimizers**：统一 CPU/GPU 内存管理，防 OOM
- **Gradient Checkpointing**：用计算换显存

### 显存估算（1.5B 模型）

| 组件 | 占用 |
|------|------|
| 基座模型 (4-bit) | ~1GB |
| LoRA 参数 (fp16) | ~4MB (r=8) |
| 优化器状态 (AdamW) | ~12MB |
| 激活值 | ~3-4GB |
| **总计** | **~5GB** |

## License

MIT
