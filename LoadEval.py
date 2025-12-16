import re
import json
import torch
from tqdm import tqdm
from peft import PeftModel
from LoadData import load_medical_dataset, INFERENCE_PROMPT_STYLE
from LoadModel import getModel
# Metrics
from sacrebleu import sentence_bleu
from evaluate import load
from transformers import AutoTokenizer
# Parallel
import torch.multiprocessing as mp

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


def extract_think(text):
    """从模型输出中找出 <think>...</think> 内容"""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    return match.group(1).strip() if match else ""

def extract_response(text):
    """从模型输出中找出 </think>...之后的 内容"""
    match = re.search(r"</think>(.*)", text, flags=re.S)
    return match.group(1).strip() if match else ""

def compute_rougeL(pred, ref):
    rouge_metric = load("/mnt/zhangchen/S3Precision/SoftLink/llm-fintune/evaluate/metrics/rouge/rouge.py")
    res = rouge_metric.compute(
        predictions=[pred],
        references=[ref],
        tokenizer=lambda x: list(x), # 中文按字切分
        use_stemmer=True
    )
    return res["rougeL"]

def compute_bleu(pred, ref):
    return sentence_bleu(pred, [ref]).score

def truncate(texts, tokenizer):
    ids = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        return_tensors=None,
    )["input_ids"]
    return tokenizer.batch_decode(ids, skip_special_tokens=True)

def compute_bert_score(preds, refs):
    bert_dir = "/mnt/zhangchen/models/BERT/xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(
        bert_dir
    )
    preds = truncate(preds, tokenizer)
    refs  = truncate(refs, tokenizer)

    bertscore = load("/mnt/zhangchen/S3Precision/SoftLink/llm-fintune/evaluate/metrics/bertscore/bertscore.py")
    results = bertscore.compute(
        predictions=preds, 
        references=refs, 
        model_type=bert_dir,
    )
    
    return results['precision']

def evalMetrics(model_name="Qwen/Qwen3-8B", device="cuda:0", 
               model=None, tokenizer=None, peft_log=None,
               dataset=None, metrics=["rougeL", "BLEU"], return_queue=None, start_index=0):
    ### Load Saved Perf Model
    if model == None:
        models_dir = "/mnt/zhangchen/models/"
        model, tokenizer = getModel(models_dir + model_name, device)
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    if peft_log != None:
        model = PeftModel.from_pretrained(model, "checkpoint/" + peft_log)
    dataLen = len(dataset['Question'])
    
    if "llama" in model_name.lower():   # Llama needs pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    Results = []
    model.eval()
    batch = 64
    for i in tqdm(range(0, dataLen, batch)):
        # Get Prediction
        questions = dataset['Question'][i: i + batch]
        inputs = tokenizer( # 'input_ids' 'attention_mask'
            [INFERENCE_PROMPT_STYLE.format(que) for que in questions],
            return_tensors="pt",
            padding=True,           # 启用填充，使批次内长度一致
        ).to(device)
        responses = runSample(model=model, tokenizer=tokenizer, device=device, inputs=inputs)
        preds_cot      = [extract_think(text)    for text in responses]
        preds_response = [extract_response(text) for text in responses]
        # Get Reference
        refs_cot       = dataset['Complex_CoT'][i: i + batch]
        refs_response  = dataset['Response'][i: i + batch]

        for j in range(len(responses)):
            record = {
                "index": start_index + i + j,
                "predictions": responses[j],
            }

            if "rougeL" in metrics:  # 动态启用 Rouge
                record["preds_cot"] = preds_cot[j]
                record["refs_cot"] = refs_cot[j]
                record["rougeL"] = compute_rougeL(
                    preds_cot[j], refs_cot[j]
                )
            if "BLEU" in metrics:   # 动态启用 BLEU
                record["preds_response"] = preds_response[j]
                record["refs_response"] = refs_response[j]
                record["BLEU"] = compute_bleu(
                    preds_response[j], refs_response[j]
                )
            Results.append(record)

        if "BERTScore" in metrics:   # 动态启用 BERTScore
            BERTScores = compute_bert_score(preds_response, refs_response)
            for j, record in enumerate(Results[i: i+batch]):
                if "preds_response" not in record:
                    record["preds_response"] = preds_response[j]
                if "refs_response" not in record:
                    record["refs_response"] = refs_response[j]
                record["BERTScore"] = BERTScores[j]

    # 返回本 GPU 结果
    return_queue.put(Results)

def evalMetrics_multi_gpu(model_name="Qwen/Qwen3-8B", 
                model=None, tokenizer=None, peft_log=None,
                dataset=None, metrics=["rougeL", "BLEU"],
                world_size=4,):
    # 给每条样本加 index，便于切分 dataset 为 world_size 份
    dataset = dataset.add_column("index", list(range(len(dataset))))
    ds_slices = []
    per_gpu = (len(dataset) + world_size - 1) // world_size
    for i in range(world_size):
        start = i * per_gpu
        end = min(start + per_gpu, len(dataset))
        ds_slices.append(dataset[start:end])

    # === 启动多进程 ===
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_queue = manager.Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=evalMetrics,
            args=(model_name, f"cuda:{rank}",
                  model, tokenizer, peft_log,
                  ds_slices[rank], metrics, return_queue, rank * per_gpu)
        )
        p.start()
        processes.append(p)
    for p in processes: p.join()    # 阻塞主进程，等待这个子进程结束 -> 等待所有 GPU 完成

    # === 收集4卡结果 ===
    all_results = []
    while not return_queue.empty():
        all_results.extend(return_queue.get())
    # === 按 index 排序 ===
    all_results = sorted(all_results, key=lambda x: x["index"])

    # === 计算汇总指标 ===
    summary = {}
    if "rougeL" in metrics:
        summary["mean_rougeL"] = round(
            sum(r["rougeL"] for r in all_results) / len(all_results), 4
        )
    if "BLEU" in metrics:
        summary["mean_BLEU"] = round(
            sum(r["BLEU"] for r in all_results) / len(all_results), 4
        )
    if "BERTScore" in metrics:
        summary["mean_BERTScore"] = round(
            sum(r["BERTScore"] for r in all_results) / len(all_results), 4
        )
    all_results.append({"summary": summary})

    # === 保存 JSON ===
    save_dir  = "" if peft_log is None else peft_log
    save_file = "checkpoint/" + f"{save_dir}/EvalResults_multiGPU.json"

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"🔥 已保存多 GPU 评测结果到: {save_file}")
    print(f"📌 汇总指标: {summary}")
    return all_results


if __name__ == "__main__":
    ### cuda
    import argparse, os
    parser = argparse.ArgumentParser(description="LLM Eval.")
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    ### Model Config
    models_dir = "/mnt/zhangchen/models/"
    model_ori, tokenizer = getModel(models_dir + "Qwen/Qwen3-8B", "cuda:0")
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    ### Load Dataset
    train_dataset = load_medical_dataset(
        dataset_path="/mnt/zhangchen/S3Precision/datasets/medical-o1-reasoning-SFT",
        eos_token=EOS_TOKEN, split="train[:-2000]", language="zh_mix", mode="train")
    eval_dataset = load_medical_dataset(
        dataset_path="/mnt/zhangchen/S3Precision/datasets/medical-o1-reasoning-SFT",
        eos_token=EOS_TOKEN, split="train[-2000:]", language="zh_mix", mode="eval")
    ### Tokenizer
    questions = eval_dataset[0:32]['Question'] 
    inputs = tokenizer( # 'input_ids' 'attention_mask'
        [INFERENCE_PROMPT_STYLE.format(que) for que in questions],
        return_tensors="pt",
        padding=True,           # 启用填充，使批次内长度一致
        truncation=True,        # 启用截断，防止序列过长
        max_length=2048         # 设置一个合适的最大长度，可根据你的模型和GPU内存调整
    ).to("cuda")

    ### Test 0
    # response = runSample(model_name="Qwen/Qwen3-8B", device="cuda:0", inputs=inputs, peft_log="baseline/epochs_1/checkpoint-730")
    # print("Inference after finetune:\n", response[0])

    ### Test 1
    # evalMetrics(model=model_ori, tokenizer=tokenizer, device="cuda:0", 
    #             inputs=inputs, peft_log="baseline/epochs_1/checkpoint-730", 
    #             dataset=eval_dataset, metrics=["rougeL"])
    # evalMetrics_multi_gpu(model=model_ori, tokenizer=tokenizer, world_size=len(args.cuda.split(",")),
    #             inputs=inputs, peft_log="baseline/epochs_1/checkpoint-730", 
    #             dataset=eval_dataset, metrics=["rougeL"])
    ### Test 2
    # evalMetrics(model=model_ori, tokenizer=tokenizer, device="cuda:0", 
    #             inputs=inputs, peft_log="baseline/epochs_1/checkpoint-730", 
    #             dataset=eval_dataset, metrics=["BLEU"])
    evalMetrics_multi_gpu(model_name="llama/Llama-3-8B-Instruct", 
                        peft_log="llama/Llama-3-8B-Instruct/te/rank_32/nvfp4/epochs_1/checkpoint-730", 
                        dataset=eval_dataset, metrics=["BLEU", "rougeL", "BERTScore"], 
                        world_size=len(args.cuda.split(",")),)

