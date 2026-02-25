import copy
import os
from tqdm import tqdm
import json
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from scipy.stats import kurtosis
import torch

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
import hf_path_config


def contribution_aware_adaptive_prune(local_lora_A, local_lora_B, global_lora_A, global_lora_B, pruning_function):
    delta_lora_A = local_lora_A - global_lora_A
    delta_lora_B = local_lora_B - global_lora_B

    norm_A_original = torch.abs(delta_lora_A).sum()
    norm_B_original = torch.abs(delta_lora_B).sum()

    norm_B_cols = torch.linalg.norm(local_lora_B, dim=0) # Shape: (r,)
    norm_A_rows = torch.linalg.norm(global_lora_A, dim=1) # Shape: (r,)

    importance_A = torch.abs(delta_lora_A) * norm_B_cols.unsqueeze(1) # unsqueeze for (r, 1) * (r, d_in) -> (r, d_in)
    importance_B = torch.abs(delta_lora_B) * norm_A_rows.unsqueeze(0) # unsqueeze for (d_out, r) * (1, r) -> (d_out, r)

    kurtosis_A = kurtosis(importance_A.view(-1).cpu().numpy(), fisher=False)
    kurtosis_B = kurtosis(importance_B.view(-1).cpu().numpy(), fisher=False)

    pruning_ratio_A = pruning_function(kurtosis_A)
    pruning_ratio_B = pruning_function(kurtosis_B)

    if pruning_ratio_A > 0:
        threshold_A = torch.quantile(importance_A, pruning_ratio_A)
        delta_lora_A[importance_A < threshold_A] = 0.0

    if pruning_ratio_B > 0:
        threshold_B = torch.quantile(importance_B, pruning_ratio_B)
        delta_lora_B[importance_B < threshold_B] = 0.0

    norm_A_pruned = torch.abs(delta_lora_A).sum()
    norm_B_pruned = torch.abs(delta_lora_B).sum()

    if norm_A_pruned > 1e-9:
        scale_A = norm_A_original / norm_A_pruned
        delta_lora_A = delta_lora_A * scale_A

    if norm_B_pruned > 1e-9:
        scale_B = norm_B_original / norm_B_pruned
        delta_lora_B = delta_lora_B * scale_B
    
    return delta_lora_A, delta_lora_B


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
save_config(script_args, fed_args)
print(script_args, fed_args)

training_args = get_training_args(script_args, script_args.learning_rate)
set_seed(script_args.seed)

if len(script_args.dataset_list) != len(script_args.dataset_sample_list):
    raise ValueError("The length of dataset_list and dataset_sample_list must be the same.")

if fed_args.num_clients != len(script_args.dataset_list):
    raise ValueError("The number of clients must be equal to the length of dataset_list.")

if script_args.template not in ['alpaca', 'vicuna',]:
    raise ValueError("If you use the official chat template, the lora config should use modules_to_save=['lm_head', 'embed_token'], which largely increase trainable params.")

# ===== Load the dataset =====
local_datasets = []
for dataset_name, dataset_sample in zip(script_args.dataset_list, script_args.dataset_sample_list):
    dataset = get_dataset(dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(dataset_name, dataset, dataset_sample)
    local_datasets.append(dataset)

sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    cache_dir = hf_path_config.HF_CACHE_DIR,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

if script_args.peft_lora_target_modules == ['all-linear']:
    all_linear_names = find_all_linear_names(model)
    peft_config.target_modules = all_linear_names
    print(f">>> Set target_modules to all linear layers: {peft_config.target_modules}")
elif script_args.peft_lora_target_modules is not None:
    peft_config.target_modules = script_args.peft_lora_target_modules
    print(f">>> Set target_modules to all linear layers: {peft_config.target_modules}")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, cache_dir = hf_path_config.HF_CACHE_DIR, use_fast=False, padding_side="right")
print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token)
print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id)
tokenizer = setup_tokenizer(tokenizer, script_args.model_name_or_path, use_official_chat=False)
print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token)
print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id)

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
print(response_template_ids, tokenizer.decode(response_template_ids))
if script_args.model_name_or_path in ['meta-llama/Llama-3.2-3B', "Qwen/Qwen2-7B", ]:
    response_template_ids = response_template_ids[1:]

data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
log_history = {cid: [] for cid in range(fed_args.num_clients)}

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
        if script_args.max_steps == -1:
            sub_dataset = local_datasets[client]
        else:
            sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        log_history[client].append({"round": round, "train_loss": results.training_loss, "sample_num": sample_num_list[client], \
                                    "metrics": results.metrics})

        # ===== Client transmits local information to server =====

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))
        delta_to_upload = {}
        lora_A_keys = [key for key in local_dict_list[client].keys() if 'lora_A.weight' in key]
        for lora_A_key in lora_A_keys:
            lora_B_key = lora_A_key.replace('lora_A.weight', 'lora_B.weight')

            if lora_B_key in local_dict_list[client]:
                pruned_A, pruned_B = contribution_aware_adaptive_prune(
                    local_dict_list[client][lora_A_key], local_dict_list[client][lora_B_key], global_dict[lora_A_key], global_dict[lora_B_key],
                    lambda k: fed_args.base + min(fed_args.upper, fed_args.coeff * math.log(max(k, 1)))
                )

                delta_to_upload[lora_A_key] = pruned_A
                delta_to_upload[lora_B_key] = pruned_B
            else:
                print(f"Warning: Found lora_A key '{lora_A_key}' but missing corresponding lora_B key.")
                raise KeyError(f"Missing corresponding lora_B key for '{lora_A_key}'")
            
        local_dict_list[client] = delta_to_upload

    # ===== Server aggregates the local models =====
    delta_global_dict = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, weighted_aggr=(not fed_args.simple_aggr)
    )
    print(f"Add delta update to previous global dict")
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] + delta_global_dict[key]

    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        print(f"Saving the model at round {round+1}...")
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-round{round+1}"))
    
    with open(os.path.join(script_args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=4, ensure_ascii=False)