import dw
import torch
import random
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
import transformer_engine.pytorch as te

def same_seed(seed=42):
    random.seed(42)
    torch.manual_seed(42)

### replace the model
def convert_linear_to_te(linear: nn.Linear):
    te_linear = te.Linear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=(linear.bias is not None),
        params_dtype=linear.weight.dtype,
    )

    # 复制权重
    te_linear.weight.data.copy_(linear.weight.data)
    if linear.bias is not None:
        te_linear.bias.data.copy_(linear.bias.data)

    return nn.ModuleDict({"default": te_linear, })

def convert_linear_to_dw(linear: nn.Linear, precision="baseline"):
    dwLinear = dw.modules.FcLayer(linear, precision)
    return nn.ModuleDict({"default": dwLinear, })

def replace_lora_modules(model, arch="te", precision="baseline"):
    for name, module in model.named_children():
        # 递归替换
        replace_lora_modules(module, arch, precision)

        # if isinstance(module, LoraLayer):
        if name in ["lora_A", "lora_B"]:
            if arch == "te":
                setattr(model, name, convert_linear_to_te(module["default"].bfloat16()))
            elif arch == "dw":
                setattr(model, name, convert_linear_to_dw(module["default"].bfloat16(), precision))
