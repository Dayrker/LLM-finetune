import torch
import random

def same_seed(seed=42):
    random.seed(42)
    torch.manual_seed(42)


