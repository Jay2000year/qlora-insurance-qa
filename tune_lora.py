"""
LoRA 超参调优实验
— 自动跑多组参数 → 每个独立 TensorBoard 日志 → 最后画对比图
"""
import os, json
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch

MODEL_PATH = "d:/llm-project/models/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
OUTPUT_DIR = "./output/lora-qwen-insurance"
MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
RESULTS_FILE = "output/tuning_results.json"

# ============================================================
# 实验矩阵
# ============================================================
EXPERIMENTS = [
    # (name,  r,   alpha,  lr,    epochs, dropout)
    ("r04-a16-lr2e4-ep3",   4,  16, 2e-4, 3, 0.05),
    ("r04-a32-lr2e4-ep3",   4,  32, 2e-4, 3, 0.05),
    ("r08-a32-lr2e4-ep3",   8,  32, 2e-4, 3, 0.05),
    ("r16-a32-lr2e4-ep3",  16,  32, 2e-4, 3, 0.05),
    ("r08-a64-lr2e4-ep3",   8,  64, 2e-4, 3, 0.05),
    ("r08-a32-lr1e4-ep3",   8,  32, 1e-4, 3, 0.05),
    ("r08-a32-lr5e4-ep3",   8,  32, 5e-4, 3, 0.05),
    ("r08-a32-lr2e4-ep5",   8,  32, 2e-4, 5, 0.05),
    ("r16-a64-lr1e4-ep5",  16,  64, 1e-4, 5, 0.10),  # 最终组合
]

def load_data():
    """加载训练数据（只加载一次，所有实验共用）"""
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    raw_data = []
    with open("d:/llm-project/data/insurance_qa.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    def format_instruction(sample):
        text = f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"
        return {"text": text}

    dataset = Dataset.from_list(raw_data).map(format_instruction)

    def tokenize(sample):
        result = tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    return tokenized


def load_fresh_model():
    """每次实验前重新加载基座模型，避免 LoRA 残留"""
    print("  Loading fresh base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def run_experiment(tokenized, name, r, alpha, lr, epochs, dropout):
    """跑一组参数，返回最终 loss"""
    timestamp = datetime.now().strftime("%m%d-%H%M")
    logging_dir = f"./output/logs/{name}-{timestamp}"

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}  (r={r}, alpha={alpha}, lr={lr}, epochs={epochs})")
    print(f"{'='*60}")

    model, tokenizer = load_fresh_model()

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, logging_dir=logging_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=epochs, learning_rate=lr,
        warmup_ratio=0.1, lr_scheduler_type="cosine",
        fp16=True, logging_steps=5, save_strategy="no",
        report_to="tensorboard", remove_unused_columns=False,
    )

    trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized)
    result = trainer.train()

    final_loss = result.training_loss
    trainable_pct = peft_model.get_nb_trainable_parameters()[0] / peft_model.num_parameters() * 100
    metrics = {
        "r": r, "alpha": alpha, "lr": lr, "epochs": epochs, "dropout": dropout,
        "trainable%": round(trainable_pct, 4),
        "final_loss": round(final_loss, 4),
        "runtime_s": round(result.metrics.get("train_runtime", 0), 1),
    }
    print(f"  Final Loss: {final_loss:.4f}  |  Trainable%: {trainable_pct:.3f}%  |  Runtime: {metrics['runtime_s']:.0f}s")
    return metrics, peft_model


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # 数据只加载一次
    tokenized = load_data()

    results = []
    best_loss = float("inf")
    best_name = None

    for name, r, alpha, lr, epochs, dropout in EXPERIMENTS:
        metrics, peft_model = run_experiment(tokenized, name, r, alpha, lr, epochs, dropout)
        results.append({"name": name, **metrics})

        if metrics["final_loss"] < best_loss:
            best_loss = metrics["final_loss"]
            best_name = name
            # 保存最佳模型
            peft_model.save_pretrained(f"./output/best_model-{name}")
            print(f"  >> 新最佳！loss={best_loss:.4f}")

        # 清显存
        del peft_model
        torch.cuda.empty_cache()

    # 保存结果
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  调优完成！最佳模型: {best_name} (loss={best_loss:.4f})")
    print(f"  结果保存: {RESULTS_FILE}")
    print(f"{'='*60}")
