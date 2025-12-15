import os
import torch
import argparse
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
# Defined modules
from LoadData import load_medical_dataset, INFERENCE_PROMPT_STYLE
from LoadModel import getModel, getLoRAModel
from LoadEval import runSample
from utils import same_seed, replace_lora_modules
# Finetune config
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
# transformer engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from contextlib import nullcontext

parser = argparse.ArgumentParser(description="LLM Inference.")
parser.add_argument("--cuda", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--train_epochs", type=int, default=1)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--arch", type=str, default="baseline")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
same_seed(42)

### Model Config
models_dir = "/mnt/zhangchen/models/"
model_ori, tokenizer = getModel(models_dir + args.model, "cuda:0")
if "llama" in args.model.lower():   # Llama needs pad_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
### LORA model
model_lora, peft_config = getLoRAModel(model_ori, args.lora_rank)   # 此时model_ori也已经被修改
# model_lora.gradient_checkpointing_enable()    # 重算激活值，减少60%显存占用，耗时增加10倍

### Load dataset
train_dataset = load_medical_dataset(
    dataset_path="/mnt/zhangchen/S3Precision/datasets/medical-o1-reasoning-SFT",
    eos_token=EOS_TOKEN, split="train[:-2000]", language="zh_mix", mode="train")
eval_dataset  = load_medical_dataset(
    dataset_path="/mnt/zhangchen/S3Precision/datasets/medical-o1-reasoning-SFT",
    eos_token=EOS_TOKEN, split="train[-2000:]", language="zh_mix", mode="eval")

### Test Before Finetune
question = train_dataset[10]['Question']
inputs = tokenizer(
    [INFERENCE_PROMPT_STYLE.format(question)] * 32,
    return_tensors="pt",
    padding="longest",
    pad_to_multiple_of=32,
).to("cuda")
# response = runSample(model=model_ori, tokenizer=tokenizer, device="cuda:0", inputs=inputs)
# print("-----------------Test Before Finetune-----------------------------")
# print(response[0].split("### Response:")[1])

### Fintune Initialize
output_dir = f"checkpoint/{args.model}/{args.arch}/rank_{args.lora_rank}/{args.precision}/epochs_{args.train_epochs}"
training_arguments = TrainingArguments( # Training Arguments
    output_dir=output_dir,
    per_device_train_batch_size=args.batch_size,  # 32B -> 8; 8B -> 32
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    num_train_epochs=args.train_epochs,
    logging_steps=50,
    logging_strategy="steps",
    save_strategy="epoch",
    save_steps=1,
    warmup_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    group_by_length=True,
    report_to="none",
)
# print("training_arguments:", training_arguments)
trainer = SFTTrainer(   # Initialize the Trainer
    model=model_lora,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, 
                    mlm=False,
                    pad_to_multiple_of=32   # For MXFP8
                    ),
)
# convert te model
if args.arch != "baseline":
    replace_lora_modules(model_lora, arch=args.arch, precision=args.precision)
print("LORA model:", model_lora)
# metrics = trainer.evaluate()
# print("first evaluate:", metrics)

### Model Training
if args.precision == "mxfp8":
    te_recipe = recipe.MXFP8BlockScaling(fp8_format=recipe.Format.HYBRID)
elif args.precision == "nvfp4":
    te_recipe = recipe.NVFP4BlockScaling(fp4_format=recipe.Format.E2M1)    # nvfp4 Only has E2M1

with te.autocast(enabled=True, recipe=te_recipe) if args.arch == "te" else nullcontext():
    trainer.train()

    # ### Model Inference After Fine-Tuning
    # response = runSample(model=model_ori, tokenizer=tokenizer, device="cuda:0", inputs=inputs)
    # print("Inference after fintune:\n", response[0].split("### Response:")[1])
    # metrics = trainer.evaluate()
    # print("last evaluate:", metrics)
