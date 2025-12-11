import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Model Config
def getModel(model_dir, cuda_maps, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        padding_side='left',
        use_fast=True)
    model_ori = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=cuda_maps,  
        dtype=dtype,    
    )
    model_ori.config.use_cache = False
    model_ori.config.pretraining_tp = 1

    return model_ori, tokenizer


# LoRA config (微调LORA设置)
def getLoRAModel(model_ori):
    peft_config = LoraConfig(
        lora_alpha=16,                           # Scaling factor for LoRA
        lora_dropout=0.05,                       # Add slight dropout for regularization
        r=64,                                    # Rank of the LoRA update matrices
        bias="none",                             # No bias reparameterization
        task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target modules for LoRA
    )
    
    return get_peft_model(model_ori, peft_config), peft_config
