import torch
from peft import PeftModel
from LoadData import load_medical_dataset, INFERENCE_PROMPT_STYLE
from LoadModel import getModel

def runSample(model_name="Qwen/Qwen3-8B", device="cuda:0", inputs=None, model=None, tokenizer=None, peft_log=None):
    ### Load Saved Perf Model
    if model == None:
        models_dir = "/mnt/zhangchen/models/"
        model, tokenizer = getModel(models_dir + model_name, device)
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    if peft_log != None:
        model = PeftModel.from_pretrained(model, "checkpoint/" + peft_log)
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    ### Model Config
    models_dir = "/mnt/zhangchen/models/"
    model_ori, tokenizer = getModel(models_dir + "Qwen/Qwen3-8B", "auto")
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    ### Load Dataset
    train_dataset = load_medical_dataset(
        dataset_path="/mnt/zhangchen/S3Precision/datasets/medical-o1-reasoning-SFT",
        eos_token=EOS_TOKEN, split="train[:-2000]", language="zh_mix", mode="train")
    eval_dataset = load_medical_dataset(
        dataset_path="/mnt/zhangchen/S3Precision/datasets/medical-o1-reasoning-SFT",
        eos_token=EOS_TOKEN, split="train[-2000:]", language="zh_mix", mode="eval")

    question = train_dataset[10]['Question'] 
    inputs = tokenizer( # 'input_ids' 'attention_mask'
        [INFERENCE_PROMPT_STYLE.format(question)],
        return_tensors="pt"
    ).to("cuda")

    response = runSample(model_name="Qwen/Qwen3-8B", device="cuda:0", inputs=inputs, peft_log="baseline/checkpoint-730")
    print("Inference after finetune:\n", response[0])
