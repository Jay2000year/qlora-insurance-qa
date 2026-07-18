"""
QLoRA 微调 Qwen2.5-1.5B — 逐行注释学习版
用法：改一个参数 → 运行 → 看 loss 变化 → 理解原理
"""

# ------------------------------------------------------------------
# 导入依赖
# ------------------------------------------------------------------
import torch                              # PyTorch 深度学习框架，提供 GPU 计算、张量操作
from transformers import (                # HuggingFace transformers 库
    AutoModelForCausalLM,                 # 自动加载因果语言模型（GPT/Qwen 这类）
    AutoTokenizer,                        # 自动加载对应的分词器
    TrainingArguments,                    # 训练配置类（lr/epoch/batch_size 等）
    Trainer,                              # 训练器，封装了训练循环、梯度更新、checkpoint
)
from peft import (                        # PEFT = Parameter-Efficient Fine-Tuning
    LoraConfig,                           # LoRA 配置类（r/alpha/target_modules 等）
    get_peft_model,                       # 把 LoRA 注入到原始模型，返回可训练的 PEFT 模型
    TaskType,                             # 任务类型枚举：CAUSAL_LM=因果语言模型
    prepare_model_for_kbit_training,      # 为 4-bit 模型做训练前准备（冻结部分参数等）
)
from datasets import Dataset              # HuggingFace datasets 库，把 dict 转成训练用的 Dataset
import json                               # 读 JSONL 数据文件
from datetime import datetime             # 时间戳，区分不同训练日志


# ==================================================================
# 一、模型加载
# ==================================================================

MODEL_PATH = "d:/llm-project/models/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
# 本地模型路径，model.safetensors（权重）、config.json（架构）、tokenizer.json 都在这里

model = AutoModelForCausalLM.from_pretrained(
    # from_pretrained：从路径加载预训练模型
    # "AutoModel" 会自动识别 config.json 里的模型类型（这里是 qwen2）
    MODEL_PATH,                           # 模型路径
    torch_dtype=torch.float16,            # 模型参数用 FP16（2 字节），比 FP32 省一半显存
    device_map="auto",                    # 自动分配模型到 GPU/CPU，单卡就是全放 GPU 0
    load_in_4bit=True,                    # 关键！用 bitsandbytes 把权重量化到 4-bit（NF4 格式）
    bnb_4bit_compute_dtype=torch.float16, # 前向/反向传播时反量化到 FP16 再算
    bnb_4bit_use_double_quant=True,       # 双重量化：量化 scale 也量化，再省 ~0.5GB
)
# 此时 model 的每一层权重都是 4-bit 存储的，不可直接训练

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# 加载分词器：把文字↔token id 互相转换
# Qwen 的 tokenizer 基于 BPE，中文一个字可能对应 1-3 个 token

tokenizer.pad_token = tokenizer.eos_token
# Qwen 的 tokenizer 没有专门的 [PAD] token（它训练时不需要）
# 我们用 EOS（结束符，<|im_end|>）来当 PAD，不影响模型理解

model = prepare_model_for_kbit_training(model)
# 为 k-bit（4-bit/8-bit）模型的训练做准备：
# 1. 冻结所有 4-bit 权重（requires_grad=False）
# 2. 把 LayerNorm/RMSNorm 层转成 FP32（这些对精度敏感，不能量化）
# 3. 启用 gradient checkpointing（省显存，用计算换内存）


# ==================================================================
# 二、LoRA 配置
# ==================================================================

lora_config = LoraConfig(
    r=4,                                   # rank：低秩分解的中间维度
    # r 越小 → 参数越少 → 训练越快 → 但拟合能力弱
    # r 越大 → 参数越多 → 表达能力强 → 但数据少时容易过拟合
    lora_alpha=32,                         # 缩放因子，实际影响力 ∝ alpha / r
    target_modules=[                       # LoRA 加到哪些层上
        "q_proj",   # Query 投影矩阵   W_Q
        "k_proj",   # Key 投影矩阵      W_K
        "v_proj",   # Value 投影矩阵    W_V
        "o_proj",   # Output 投影矩阵   W_O
    ],
    # 这些名字来自 Qwen 模型内部的层名，不同模型名字不同（LLaMA 叫 q_proj，GPT-2 叫 c_attn）
    lora_dropout=0.05,                     # LoRA 参数在前向时随机丢弃 5%，防过拟合
    bias="none",                           # 不训练 bias 参数
    task_type=TaskType.CAUSAL_LM,          # 因果语言模型任务（GPT/Qwen 都是这个）
)

model = get_peft_model(model, lora_config)
# 把 LoRA 注入到模型中：
# 原始权重 W 冻结不动，在旁边挂两个小矩阵 A 和 B
# 前向时：output = W·x + (alpha/r) · (A·B)·x
# 只有 A 和 B 参与训练

model.print_trainable_parameters()
# 打印：trainable params: X || all params: Y || trainable%: Z%
# 你应该看到 trainable% 在 0.05% ~ 1% 之间


# ==================================================================
# 三、数据构造
# ==================================================================

def format_instruction(sample):
    """把 {'instruction':..., 'output':...} 转成 Qwen 对话格式的字符串"""
    text = (
        f"<|im_start|>user\n"              # 用户消息开始标记
        f"{sample['instruction']}"          # 用户问题
        f"<|im_end|>\n"                    # 用户消息结束
        f"<|im_start|>assistant\n"         # 助手消息开始标记
        f"{sample['output']}"              # 期望的助手回答
        f"<|im_end|>"                      # 助手消息结束
    )
    return {"text": text}
    # 返回一个 dict，key 是 "text"，value 是一整段对话文本

raw_data = []
with open("d:/llm-project/data/insurance_qa.jsonl", "r", encoding="utf-8") as f:
    for line in f:                         # 逐行读 JSONL 文件
        raw_data.append(json.loads(line))   # 每行是一个 JSON: {"instruction":"...","output":"..."}
# 读完后 raw_data = [{"instruction":"...","output":"..."}, {...}, ...] 共 100 条

# raw_data = raw_data[:20]  # 取消注释可以只用 20 条数据，跑得更快

dataset = Dataset.from_list(raw_data)
# 把 Python list 转成 HuggingFace Dataset 对象
# Dataset 支持 .map() 方法做批量处理，比 for 循环快

dataset = dataset.map(format_instruction)
# 对每一条数据调用 format_instruction，把 instruction/output 拼成对话文本
# 结果 dataset 里每条数据多了一个 "text" 字段："<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"


# ==================================================================
# 四、Tokenize（文本 → 数字）
# ==================================================================

MAX_LENGTH = 512  # 最大序列长度（token 数），超过的被截断，不够的补 PAD

def tokenize(sample):
    result = tokenizer(
        sample["text"],                    # 输入："<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n..."
        truncation=True,                   # 超过 max_length 的截掉
        max_length=MAX_LENGTH,             # 统一补到 512
        padding="max_length",              # 把每个序列都补齐到 512
        # padding="max_length" 意味着：短序列后面加 [PAD]，长序列截断
        # 也可以设 "longest"（补到 batch 内最长），但需要额外 collator
    )
    result["labels"] = result["input_ids"].copy()
    # labels 是训练时的正确答案，模型会根据 labels 算 loss
    # 这里 labels = input_ids → 每个 token 都参与 loss 计算（包括 instruction 部分）
    # 更好的做法是把 instruction 部分的 labels 设为 -100（忽略），让模型只学回答
    return result

tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names    # 删掉原始列（instruction/output/text），只保留 token 化结果
)
# tokenized_dataset 每条数据包含：
# - input_ids: [t1, t2, ..., t512]  输入 token 序列
# - attention_mask: [1, 1, ..., 0]  1=真实token, 0=PAD（不参与 attention 计算）
# - labels: [t1, t2, ..., t512]     监督信号（目前和 input_ids 相同）


# ==================================================================
# 五、训练参数
# ==================================================================

training_args = TrainingArguments(
    output_dir="./output/lora-qwen-insurance",  # 模型和 checkpoint 保存位置
    logging_dir=f"./output/logs/r4-a32-lr2e4-ep3-{datetime.now():%m%d-%H%M}",  # 每次训练独立日志
    per_device_train_batch_size=2,   # 每张 GPU 每次吃 2 条数据
    gradient_accumulation_steps=4,   # 攒 4 步的梯度再一起更新参数
    # 有效 batch_size = 2 × 4 = 8（等效于一次吃 8 条，但不占 8 条的显存）
    num_train_epochs=3,              # 训练 3 轮（整个数据集过 3 遍）
    learning_rate=2e-4,              # 学习率：LoRA 用 1e-4 到 5e-4
    # 比全量微调（1e-5）高 10 倍，因为 LoRA 参数是随机初始化的
    warmup_ratio=0.1,                # 前 10% 步 lr 从 0 线性涨到 2e-4，避免训练初期震荡
    lr_scheduler_type="cosine",      # lr 按余弦曲线衰减：前期高、后期低，比 linear 更平滑
    fp16=True,                       # 混合精度：计算用 FP16，关键值存 FP32，省显存 + 加速
    logging_steps=5,                 # 每 5 步打印一次 loss、lr、grad_norm
    save_strategy="epoch",           # 每个 epoch 结束时保存 checkpoint
    report_to="tensorboard",          # TensorBoard 免费本地可视化
    remove_unused_columns=False,     # 保留所有列，不自动删除
)

trainer = Trainer(
    model=model,                      # 要训练的模型（已注入 LoRA）
    args=training_args,               # 训练参数
    train_dataset=tokenized_dataset,  # 训练数据
    # data_collator 没指定 → 用默认的，因为我们已经做了 padding，不需要额外处理
)
# Trainer 内部封装了：
#   训练循环（for batch in dataloader）
#   → 前向传播（model(batch) → loss）
#   → 反向传播（loss.backward() → gradients）
#   → 梯度累积（攒够 accumulation_steps 步）
#   → 参数更新（optimizer.step() + lr_scheduler.step()）
#   → 日志打印

print("\n开始训练...")
trainer.train()

# 开始训练，输出类似：
# {'loss': 5.23, 'grad_norm': 14.8, 'learning_rate': 1.8e-4, 'epoch': 0.4}
# {'loss': 0.62, 'grad_norm': 0.62, 'learning_rate': 1.5e-4, 'epoch': 0.8}
# ...
# loss 不断下降 = 模型在学习


# ==================================================================
# 六、保存模型
# ==================================================================

OUTPUT_DIR = "./output/lora-qwen-insurance"
model.save_pretrained(OUTPUT_DIR)
# 保存 LoRA 权重（adapter_model.safetensors ~17MB）和配置（adapter_config.json ~1KB）
# 注意：不保存基座模型！使用时需要加载基座 + LoRA adapter

tokenizer.save_pretrained(OUTPUT_DIR)
# 保存 tokenizer 文件（tokenizer.json、vocab.json、merges.txt 等）


# ==================================================================
# 七、测试
# ==================================================================

print("\n--- 测试 ---")
model.eval()
# 切换到评估模式，关闭 Dropout（训练时随机丢 5% 的 LoRA 参数，推理时不丢）

test_prompts = [
    "车险理赔需要什么材料？",
    "保险理赔被拒了怎么办？",
    "重疾险和医疗险有什么区别？",
]

for prompt in test_prompts:
    inputs = tokenizer(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        # 构造和训练时一样的 prompt 格式，提醒模型"现在该你回答了"
        return_tensors="pt"             # 返回 PyTorch tensor 格式
    ).to("cuda")                        # 把 tensor 挪到 GPU 上

    outputs = model.generate(
        **inputs,                        # 拆包：input_ids, attention_mask
        max_new_tokens=200,              # 最多生成 200 个新 token
        temperature=0.7,                 # 温度控制随机性：<1 更保守（选高概率词），>1 更跳跃
        do_sample=True,                  # 采样模式（非 greedy decode），让回答不千篇一律
    )
    # outputs 是一个 tensor，shape 为 (1, 512+200)，包含输入+生成的完整 token 序列

    result = tokenizer.decode(
        outputs[0],                      # 取第一个样本（batch_size=1 所以只有 outputs[0]）
        skip_special_tokens=True         # 跳过 <|im_start|> 等特殊 token，只显示文字
    )

    assistant_part = result.split("assistant")[-1]
    # "assistant" 之后的部分就是模型的回答，前面的 "user: ... assistant: " 不要

    print(f"\nQ: {prompt}")
    print(f"A: {assistant_part[:300]}")   # 最多显示 300 字
    print("-" * 50)
