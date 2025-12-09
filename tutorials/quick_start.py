from trl import SFTTrainer
from datasets import load_dataset
import os
import argparse

parser = argparse.ArgumentParser(description="LLM Inference.")
parser.add_argument("--cuda", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="/mnt/zhangchen/models/Qwen/Qwen3-8B",
    train_dataset=dataset,
)
print("train_log:", type(trainer.args))
trainer.train()