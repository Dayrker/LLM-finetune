import torch
from safetensors.torch import load_file, save_file

def convert_lora_state_dict_remove_raw_module(state_dict):
    """
    将 LoRA state_dict 中的 '.raw_module.' 去掉
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace(".default.raw_module.", ".")
        new_state_dict[new_k] = v
    return new_state_dict

if __name__ == "__main__":
    Dir = "/mnt/zhangchen/S3Precision/LLM-finetune/checkpoint/" \
        + "llama/Llama-3-8B-Instruct/dw/rank_32/mxfp8/epochs_1/checkpoint-730/"
    
    # 1. 读取原始 adapter_model.safetensors
    state_dict = load_file(Dir + "adapter_model.safetensors")

    # 2. 转换 key
    state_dict = convert_lora_state_dict_remove_raw_module(state_dict)

    # 3. 保存新的 adapter
    save_file(state_dict, Dir + "adapter_model.safetensors")
