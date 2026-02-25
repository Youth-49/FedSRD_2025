from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime, timedelta


# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=50, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    dare_p: Optional[float] = field(default=0., metadata={"help": "dropout ratio for DARE method"})
    dare_mask_strategy: Optional[str] = field(default="random", metadata={"help": "mask strategy for DARE method"})
    base: Optional[float] = field(default=0.85, metadata={"help": "the pruning function: base + min(upper, coeff*log(k))"})
    coeff: Optional[float] = field(default=0.1, metadata={"help": "the pruning function: base + min(upper, coeff*log(k))"})
    upper: Optional[float] = field(default=0.11, metadata={"help": "the pruning function: base + min(upper, coeff*log(k))"})
    download_sparse_ratio: Optional[float] = field(default=0.9, metadata={"help": "the sparsity ratio for downloading the model updates"})
    simple_aggr: Optional[bool] = field(default=False, metadata={"help": "whether to use weighted aggregation"})
    save_model_freq: Optional[int] = field(default=1, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_list: Optional[List[str]] = field(default=None, metadata={"help": "dataset name list"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "enable fp16 precision"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "enable bf16 precision"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    logging_dir: Optional[str] = field(default="output", metadata={"help": "the log directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_target_modules: Optional[List[str]] = field(default=None, metadata={"help": "the targets modules of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=100, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    dataset_sample_list: Optional[List[int]] = field(default=None, metadata={"help": "number of samples per dataset "})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})

parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

# ===== Define the LoraConfig =====
if script_args.use_peft:
    if script_args.model_name_or_path in ["meta-llama/Llama-3.2-3B", ]:
        if script_args.template in ['alpaca', 'vicuna']:
            peft_config = LoraConfig(
                r=script_args.peft_lora_r,
                lora_alpha=script_args.peft_lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            raise ValueError(f"Must specify the modules_to_save=['lm_head', 'embed_token']")
    elif script_args.model_name_or_path in ["Qwen/Qwen2-7B", ]:
        if script_args.template in ['alpaca', 'vicuna']:
                peft_config = LoraConfig(
                r=script_args.peft_lora_r,
                lora_alpha=script_args.peft_lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            raise ValueError(f"Must specify the modules_to_save=['lm_head', 'embed_token']")
    else:
        raise ValueError(f"Please specify the peft_config for {script_args.model_name_or_path}")
else:
    peft_config = None

def get_config():
    return script_args, fed_args, peft_config

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        logging_dir=script_args.logging_dir,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True}, # not a keyword argument in transformers 4.31.0
        lr_scheduler_type="constant",
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.float16
    elif script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.float16
    else:
        device_map = 'auto'
        quantization_config = None
        torch_dtype = torch.bfloat16

    return device_map, quantization_config, torch_dtype

def save_config(script_args, fed_args):
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    # dataset_name_split = os.path.basename(script_args.dataset_name)
    model_name = script_args.model_name_or_path.split('/')[-1]
    num_datasets = len(script_args.dataset_list)
    if fed_args.fed_alg in ['fedsrd', 'fedsrd-e', ]:
        output_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_p{fed_args.download_sparse_ratio}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
        logging_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_p{fed_args.download_sparse_ratio}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
    else:
        output_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
        logging_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
        

    while True:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            if fed_args.fed_alg in ['fedsrd', 'fedsrd-e', ]:
                output_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_p{fed_args.download_sparse_ratio}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
                logging_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_p{fed_args.download_sparse_ratio}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
            else:
                output_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
                logging_dir = f"{script_args.output_dir}/{num_datasets}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{model_name}_{script_args.learning_rate}_{now_time}"
                
    script_args.output_dir = output_dir
    script_args.logging_dir = logging_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        json.dump(combined_dict, f, indent=4)