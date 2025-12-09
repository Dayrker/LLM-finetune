# datasetLoad.py
# -------------------------------------------------------
# Utility for loading and formatting Medical SFT datasets
# -------------------------------------------------------

from datasets import load_dataset

# ===========================
# Prompt Templates
# ===========================

TRAIN_PROMPT_STYLE = """以下是描述一项任务的说明，以及提供进一步背景信息的输入内容。请给出恰当的回答以完成请求。
在回答之前，请仔细思考问题，并构建一个逐步的思维链，以确保回答合乎逻辑且准确无误。

### Instruction:
您是一位在临床推理、诊断和治疗规划方面拥有高级知识的医学专家。
请回答以下医学问题。

### Quesetion: {}
### Response:
<think>
{}
</think>
{}"""   # 最后的 {} 放 response（含 EOS）


INFERENCE_PROMPT_STYLE = """以下是描述一项任务的说明，以及提供进一步背景信息的输入内容。请给出恰当的回答以完成请求。
在回答之前，请仔细思考问题，并构建一个逐步的思维链，以确保回答合乎逻辑且准确无误。

### Instruction:
您是一位在临床推理、诊断和治疗规划方面拥有高级知识的医学专家。
请回答以下医学问题。

### Question: {}
### Response:
<think>"""   # 只用于推理，不包含 response


# ===========================
# Core Formatting Function
# ===========================

def _format_examples(examples, prompt_style, include_response=True, eos_token=None):
    """
    通用格式化函数，用于 dataset.map
    include_response=True  -> 训练
    include_response=False -> 推理/评估
    """
    questions = examples["Question"]
    complex_cots = examples["Complex_CoT"]
    responses = examples["Response"]

    formatted_texts = []

    for question, cot, response in zip(questions, complex_cots, responses):
        if include_response:
            text = prompt_style.format(question, cot, response)
            if eos_token is not None and not response.endswith(eos_token):  # Train是完整的句子，才需要eos
                text = text + eos_token
        else:
            text = prompt_style.format(question)

        formatted_texts.append(text)

    return {"text": formatted_texts}


# ===========================
# Public Dataset Loaders
# ===========================

def load_medical_dataset(dataset_path, eos_token, split="train[:-2000]", language="en", mode="train"):
    """加载用于训练的 SFT 数据集"""
    dataset = load_dataset(
        dataset_path,
        language,
        split=split,
        # split="train[:-2000]" if mode == "train" else "train[-2000:]",
    )


    dataset = dataset.map(
        lambda ex: _format_examples(
            ex,
            TRAIN_PROMPT_STYLE if mode == "train" else INFERENCE_PROMPT_STYLE,
            include_response = True if mode == "train" else False,
            eos_token = eos_token if mode == "train" else None,
        ),
        batched=True,
    )

    return dataset
