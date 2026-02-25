import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_tokenizer(tokenizer, model_name_or_path, use_official_chat=False):
    if tokenizer.pad_token is None:
        if model_name_or_path in ['meta-llama/Llama-3.2-3B', ]:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"  # following https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/166
            # another choice: https://github.com/lm-sys/FastChat/issues/3266#issuecomment-2083674781

    if use_official_chat:
        if model_name_or_path in ['meta-llama/Llama-3.2-3B', ]:
            tokenizer.eos_token = "<|eot_id|>"
        elif model_name_or_path in ["Qwen/Qwen2-7B", ]:
            tokenizer.eos_token = "<|im_end|>"

    return tokenizer


import torch
def find_all_linear_names(model):
    """
    Finds all linear layer names in a given model, including those
    from bitsandbytes if available.
    """
    linear_classes = (torch.nn.Linear,)
    try:
        import bitsandbytes as bnb
        linear_classes += (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
        print("Found bitsandbytes, including its linear layers.")
    except ImportError:
        print("bitsandbytes not found, only searching for torch.nn.Linear.")
        pass

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:
        print("Excluding 'lm_head' from LoRA targets.")
        lora_module_names.remove('lm_head')
        
    print(f"Found linear layers for LoRA: {list(lora_module_names)}")
    return list(lora_module_names)
