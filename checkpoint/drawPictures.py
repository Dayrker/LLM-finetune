import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_trainer_state(path):
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state["log_history"]

def extract_series(log_history, key):
    steps = []
    values = []
    for entry in log_history:
        if key in entry and "step" in entry:
            steps.append(entry["step"])
            values.append(entry[key])
    return steps, values

if __name__ == "__main__":
    # ===== 配置你要对比的 trainer_state.json 路径 =====
    Dir = "/mnt/zhangchen/S3Precision/LLM-finetune/checkpoint/"
    model = "Qwen/Qwen3-8B"
    state = "/epochs_1/checkpoint-730/trainer_state.json"

    trainer_states = {
        "baseline": Dir + model + "/baseline/rank_32/bf16" + state,
        "mxfp8":    Dir + model + "/te/rank_32/mxfp8"      + state,
        "nvfp4":    Dir + model + "/te/rank_32/nvfp4"      + state,
        # 可以继续加
    }

    metric = "loss"  # 可改成 learning_rate / grad_norm / mean_token_accuracy / entropy

    plt.figure()
    for name, path in trainer_states.items():
        log_history = load_trainer_state(path)
        steps, values = extract_series(log_history, metric)
        plt.plot(steps, values, label=name)

    plt.xlabel("Global Step")
    plt.ylabel(metric)
    plt.title(f"Trainer State Comparison: {metric}")
    plt.legend()
    plt.savefig("loss_curve.png", dpi=400, bbox_inches="tight")