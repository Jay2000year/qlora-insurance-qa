"""
训练对比画图脚本
— 自动扫 output/logs/ 下所有子目录，读 TensorBoard 日志
— 生成 loss / learning_rate / grad_norm 三张对比图
— 每次训练独立日志 + 不同参数 → 一张图里多条线对比
"""
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LOGS_DIR = Path("output/logs")
PLOTS_DIR = Path("output/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 扫描所有日志目录
# ============================================================
runs = {}
for subdir in sorted(LOGS_DIR.iterdir()):
    if not subdir.is_dir():
        continue
    event_files = list(subdir.glob("events.out.tfevents.*"))
    if not event_files:
        continue
    runs[subdir.name] = str(event_files[0])

if not runs:
    print("未找到 TensorBoard 日志，请先跑训练脚本。")
    exit(1)

print(f"找到 {len(runs)} 个训练日志：")
for name in runs:
    print(f"  - {name}")

# ============================================================
# 2. 读取所有日志的 scalar 数据
# ============================================================
data = {}  # {run_name: {"train/loss": [(step, value), ...], ...}}

for run_name, event_path in runs.items():
    ea = EventAccumulator(event_path)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    run_data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        run_data[tag] = [(e.step, e.value) for e in events]
    data[run_name] = run_data
    print(f"读取 {run_name}: {list(run_data.keys())}")

# ============================================================
# 3. 画对比图
# ============================================================
METRICS = [
    ("train/loss",        "Training Loss",          "loss_comparison.png"),
    ("train/learning_rate", "Learning Rate",         "learning_rate_comparison.png"),
    ("train/grad_norm",   "Gradient Norm",           "grad_norm_comparison.png"),
]

for tag, ylabel, filename in METRICS:
    fig, ax = plt.subplots(figsize=(12, 6))

    for run_name, run_data in sorted(data.items()):
        if tag in run_data:
            steps = [s for s, v in run_data[tag]]
            values = [v for s, v in run_data[tag]]
            ax.plot(steps, values, marker=".", markersize=3, linewidth=1.5, label=run_name)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} — 多参数对比", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    save_path = PLOTS_DIR / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存: {save_path}")

# ============================================================
# 4. 汇总图（三合一）
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

for ax, (tag, ylabel, _) in zip(axes, METRICS):
    for run_name, run_data in sorted(data.items()):
        if tag in run_data:
            steps = [s for s, v in run_data[tag]]
            values = [v for s, v in run_data[tag]]
            ax.plot(steps, values, marker=".", markersize=2, linewidth=1.5, label=run_name)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Step", fontsize=12)
fig.suptitle("训练总览 — 多参数对比", fontsize=16, fontweight="bold")
fig.tight_layout()

summary_path = PLOTS_DIR / "training_summary.png"
fig.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] 保存汇总图: {summary_path}")
print(f"\n所有图片保存在: {PLOTS_DIR.absolute()}")
