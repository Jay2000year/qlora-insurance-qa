"""
查看 QLoRA 量化对比 + 显存/参数量计算
"""
import torch
from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
import os

MODEL_PATH = "d:/llm-project/models/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master"

print("=" * 60)
print("一、参数量计算")
print("=" * 60)

# 只看配置文件，不加载模型
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_PATH)
hidden_size = config.hidden_size       # 隐藏层维度
num_layers = config.num_hidden_layers  # 层数
num_heads = config.num_attention_heads
intermediate_size = config.intermediate_size
vocab_size = config.vocab_size

print(f"hidden_size (d_model): {hidden_size}")
print(f"num_layers: {num_layers}")
print(f"num_heads: {num_heads}")
print(f"intermediate_size: {intermediate_size}")
print(f"vocab_size: {vocab_size}")

# 粗略估算参数量
# 每层 Transformer = Attention(Q/K/V/O) + FFN(gate/up/down) + 2×RMSNorm
# Attention: 4 × (hidden × hidden) = 4 × 1536²
# FFN: 3 × (hidden × intermediate) = 3 × 1536 × 8960 (Qwen 有 SwiGLU)
# 实际数字从 config 拿

# Qwen 用的是 SwiGLU，intermediate_size 的计算
# 标准 FFN: W1(h, 4h) + W2(4h, h) = 2 × h × 4h
# SwiGLU FFN: gate(h, 8h/3) + up(h, 8h/3) + down(8h/3, h) = 3 × h × (8h/3) ≈ 8h²
# 但 Qwen2.5 的 intermediate_size 已经定义好了

attn_params_per_layer = 4 * hidden_size * hidden_size  # Q/K/V/O
ffn_params_per_layer = 3 * hidden_size * intermediate_size  # gate/up/down
norm_params_per_layer = 2 * hidden_size  # 2 个 RMSNorm per layer

total_attn = attn_params_per_layer * num_layers
total_ffn = ffn_params_per_layer * num_layers
total_norm = norm_params_per_layer * num_layers
total_embedding = vocab_size * hidden_size  # token embedding
total_lm_head = vocab_size * hidden_size  # lm_head（Qwen 不共享 embedding 和 lm_head）

total_params = total_attn + total_ffn + total_norm + total_embedding + total_lm_head

print(f"\n每层参数量:")
print(f"  Attention (Q/K/V/O): {attn_params_per_layer/1e6:.1f}M")
print(f"  FFN (gate/up/down): {ffn_params_per_layer/1e6:.1f}M")
print(f"  Total per layer: {total_attn/1e6 + total_ffn/1e6:.1f}M")
print(f"\n总参数量:")
print(f"  Attention: {total_attn/1e9:.2f}B")
print(f"  FFN: {total_ffn/1e9:.2f}B")
print(f"  Embedding + LM Head: {2*total_embedding/1e9:.2f}B")
print(f"  总计: {total_params/1e9:.2f}B")

# ============================================================
print(f"\n{'=' * 60}")
print("二、显存占用计算")
print("=" * 60)

# FP16: 每个参数 2 字节
fp16_weight_mb = total_params * 2 / (1024**2)
# 4-bit: 每个参数 0.5 字节
q4_weight_mb = total_params * 0.5 / (1024**2)
# 8-bit: 每个参数 1 字节

print(f"FP16 权重: {fp16_weight_mb:.0f} MB (~{fp16_weight_mb/1024:.1f} GB)")
print(f"8-bit 量化: {total_params * 1 / (1024**2):.0f} MB (~{total_params * 1 / (1024**3):.1f} GB)")
print(f"4-bit 量化: {q4_weight_mb:.0f} MB (~{q4_weight_mb/1024:.1f} GB)")

# 训练时额外显存
# 优化器状态 (AdamW): 每个可训练参数需要 8 字节 (fp32 param + fp32 momentum + fp32 variance)
# 但 LoRA 只训练 0.28% 参数
# 激活值: batch_size × seq_len × hidden_size × num_layers × 34 (粗略)
# 梯度: 和可训参数量相同

# 全量微调
full_optimizer_mb = total_params * 8 / (1024**2)
# LoRA 微调
lora_trainable = total_params * 0.0028
lora_optimizer_mb = lora_trainable * 8 / (1024**2)
# 梯度 (LoRA)
lora_gradient_mb = lora_trainable * 2 / (1024**2)

# 激活值 (粗略)
batch_size = 2
seq_len = 512
activation_mb = batch_size * seq_len * hidden_size * num_layers * 34 * 2 / (1024**2)  # fp16

print(f"\n训练时额外开销:")
print(f"  全量微调 - 优化器状态: {full_optimizer_mb:.0f} MB")
print(f"  LoRA 微调 - 优化器状态: {lora_optimizer_mb:.0f} MB")
print(f"  LoRA 微调 - 梯度: {lora_gradient_mb:.0f} MB")
print(f"  激活值 (估算, bs=2, seq=512): {activation_mb:.0f} MB")

# 总显存
fp16_total = fp16_weight_mb + full_optimizer_mb + activation_mb
q4_lora_total = q4_weight_mb + lora_optimizer_mb + lora_gradient_mb + activation_mb

print(f"\n总显存估算:")
print(f"  FP16 full fine-tune: {fp16_total:.0f} MB = {fp16_total/1024:.1f} GB {'!!OOM 8GB!!' if fp16_total > 8192 else 'OK on 8GB'}")
print(f"  QLoRA (4-bit): {q4_lora_total:.0f} MB = {q4_lora_total/1024:.1f} GB {'OK on 8GB' if q4_lora_total < 8192 else '!!OOM!!'}")

# ============================================================
print(f"\n{'=' * 60}")
print("三、实际加载模型，看量化前后权重对比")
print("=" * 60)

# 3.1 加载 FP16 模型
print("\n[1] 加载 FP16 模型...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",  # 先放 CPU，不然 GPU 装两个模型
)

# 取第一层 Q 投影权重
q_proj_fp16 = model_fp16.model.layers[0].self_attn.q_proj.weight.data.clone()
print(f"FP16 Q-proj 权重:")
print(f"  shape: {q_proj_fp16.shape}")
print(f"  dtype: {q_proj_fp16.dtype}")
print(f"  min={q_proj_fp16.min().item():.4f}, max={q_proj_fp16.max().item():.4f}, mean={q_proj_fp16.mean().item():.4f}")
print(f"  前 5 个值: {q_proj_fp16[0, :5].tolist()}")
print(f"  内存: {q_proj_fp16.numel() * 2 / 1024:.0f} KB (FP16)")

# 释放 FP16 模型
del model_fp16
torch.cuda.empty_cache()

# 3.2 加载 4-bit 模型
print("\n[2] 加载 4-bit 模型...")
model_q4 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 4-bit 权重在 GPU 上，需要特殊方式查看
# bitsandbytes 把权重存在 Params4bit 对象里
q_proj_4bit = model_q4.model.layers[0].self_attn.q_proj.weight

print(f"4-bit Q-proj 权重:")
print(f"  实际存储: Params4bit 对象")
print(f"  shape: {q_proj_4bit.shape}")
print(f"  存储数据量: {q_proj_4bit.numel() * 0.5 / 1024:.0f} KB (4-bit)")
print(f"  压缩比: {q_proj_4bit.numel() * 2 / (q_proj_4bit.numel() * 0.5):.1f}x")

# 反量化回 FP16 看精度损失
q4_dequant = q_proj_4bit.dequantize()
print(f"  反量化后: dtype={q4_dequant.dtype}, min={q4_dequant.min().item():.4f}, max={q4_dequant.max().item():.4f}")
print(f"  前 5 个值: {q4_dequant[0, :5].tolist()}")

# 计算量化误差
# 重新加载 FP16 到 CPU 对比
del model_q4
torch.cuda.empty_cache()

print("\n[3] 计算量化误差...")
model_ref = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="cpu"
)
q_ref = model_ref.model.layers[0].self_attn.q_proj.weight.data

# 加载 4-bit
model_q4_2 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="cpu",
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
q_q4_dequant = model_q4_2.model.layers[0].self_attn.q_proj.weight.dequantize().float()

# 误差
diff = (q_ref.float() - q_q4_dequant.float()).abs()
print(f"  最大绝对误差: {diff.max().item():.6f}")
print(f"  平均绝对误差: {diff.mean().item():.6f}")
print(f"  相对误差: {diff.mean().item() / q_ref.float().abs().mean().item() * 100:.2f}%")

del model_ref, model_q4_2

# ============================================================
print(f"\n{'=' * 60}")
print("四、显存实测（调用 nvidia-smi）")
print("=" * 60)

# Load 4-bit to GPU
print("\n加载 4-bit 模型到 GPU...")
model_gpu = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
used, total_mem = result.stdout.strip().split(", ")
print(f"4-bit 加载后显存: {used} MB / {total_mem} MB")
model_weight_mb = total_params * 0.5 / (1024**2)
print(f"  模型权重约占: {model_weight_mb:.0f} MB")
print(f"  其余为 CUDA 上下文 + KV Cache 预留: {int(used) - model_weight_mb:.0f} MB")

del model_gpu
torch.cuda.empty_cache()

# Load FP16 to GPU
print("\n加载 FP16 模型到 GPU...")
model_gpu_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
)
result2 = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
used2, _ = result2.stdout.strip().split(", ")
print(f"FP16 加载后显存: {used2} MB / {total_mem} MB")
fp16_model_mb = total_params * 2 / (1024**2)
print(f"  模型权重约占: {fp16_model_mb:.0f} MB")
print(f"  4-bit saves: {int(used2) - int(used)} MB ({(1 - int(used)/int(used2)) * 100:.0f}%)")

del model_gpu_fp16
torch.cuda.empty_cache()

print("=" * 60)
print("Summary")
print("=" * 60)
print(f"""
Total params: {total_params/1e9:.2f}B
FP16 weight: {total_params*2/1024**3:.1f} GB
4-bit weight: {total_params*0.5/1024**3:.1f} GB
Quantization error: ~{diff.mean().item() / q_ref.float().abs().mean().item() * 100:.1f}%

Why 1.5B doesn't OOM in FP16:
  FP16 weights 3.5GB + activations ~3GB = ~6.5GB < 8GB
  But FP16 FULL training: weights + optimizer(4x) + gradients = ~21GB -> OOM

Why 7B would OOM:
  FP16 weights 14GB already > 8GB
  Must use 4-bit: weights 3.5GB + activations ~5GB = ~8.5GB (barely fits)

Best for RTX 4060 8GB:
  Qwen2.5-1.5B: 4-bit or FP16 inference both OK
  Qwen2.5-7B: MUST use 4-bit QLoRA for training
  FP16 training of ANY model: need optimizer memory = 4x model -> only tiny models
""")
