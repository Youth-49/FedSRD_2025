import random
import torch
import math

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, weighted_aggr=True):
    if not weighted_aggr:
        for client in clients_this_round:
            sample_num_list[client] = 1
            
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    
    if fed_args.fed_alg in ['fedsrd',]:
        sparse_delta_to_send = {}
        lora_A_keys = [key for key in global_dict.keys() if 'lora_A' in key]
        for key_A in lora_A_keys:
            key_B = key_A.replace('lora_A', 'lora_B')
            print(f"  -> Aggregating layer: {key_A.split('.lora_A')[0]}")
            lora_B_shape = global_dict[key_B].shape
            lora_A_shape = global_dict[key_A].shape
            r = lora_A_shape[0]
            d_out, d_in = lora_B_shape[0], lora_A_shape[1]
            avg_w_local = torch.zeros((d_out, d_in), device=global_dict[key_A].device)
            for client in clients_this_round:
                lora_A_local = global_dict[key_A] + local_dict_list[client][key_A]
                lora_B_local = global_dict[key_B] + local_dict_list[client][key_B]
                w_local = lora_B_local @ lora_A_local
                avg_w_local += w_local * (sample_num_list[client] / sample_this_round)
            try:
                U, S, Vh = torch.linalg.svd(avg_w_local)
            except torch.linalg.LinAlgError as e:
                print(f"SVD failed for layer {key_A.split('.lora_A')[0]}: {e}. Skipping update for this layer.")
                raise ValueError()

            U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :] 
            s_sqrt_diag = torch.diag(torch.sqrt(S_r))
            new_lora_B = (U_r @ s_sqrt_diag).to(global_dict[key_B].dtype)
            new_lora_A = (s_sqrt_diag @ Vh_r).to(global_dict[key_A].dtype)

            B_old = global_dict[key_B]
            A_old = global_dict[key_A]
            W_old = global_dict[key_B] @ global_dict[key_A]
            W_new = new_lora_B @ new_lora_A
            delta_w_target = W_new - W_old

            # Alternating based on the round index
            if round_idx % 2 == 0:
                print(f"  -> Round {round_idx} (even): Updating LoRA A")
                B_old_pinv = torch.linalg.pinv(B_old.float()).to(B_old.dtype)
                delta_A_proj = B_old_pinv @ delta_w_target
                # Sparsify the result with random mask
                sparse_delta_A = mask_input_with_mask_rate(input_tensor=delta_A_proj, mask_rate=fed_args.download_sparse_ratio, rescale=True, mask_strategy='random')
                sparse_delta_to_send[key_A] = sparse_delta_A
                sparse_delta_to_send[key_B] = torch.zeros_like(B_old)
            else:
                print(f"  -> Round {round_idx} (odd): Updating LoRA B")
                A_old_pinv = torch.linalg.pinv(A_old.float()).to(A_old.dtype)
                delta_B_proj = delta_w_target @ A_old_pinv
                # Sparsify the result with random mask
                sparse_delta_B = mask_input_with_mask_rate(input_tensor=delta_B_proj, mask_rate=fed_args.download_sparse_ratio, rescale=True, mask_strategy='random')
                sparse_delta_to_send[key_A] = torch.zeros_like(A_old)
                sparse_delta_to_send[key_B] = sparse_delta_B
            
        return sparse_delta_to_send
    
    elif fed_args.fed_alg in ['fedsrd-e', ]:
        sparse_delta_to_send = {}
        lora_A_keys = [key for key in global_dict.keys() if 'lora_A' in key]
        for key_A in lora_A_keys:
            key_B = key_A.replace('lora_A', 'lora_B')
            print(f"  -> Aggregating layer: {key_A.split('.lora_A')[0]}")
            lora_B_shape = global_dict[key_B].shape
            lora_A_shape = global_dict[key_A].shape
            r = lora_A_shape[0]
            d_out, d_in = lora_B_shape[0], lora_A_shape[1]
            avg_w_local = torch.zeros((d_out, d_in), device=global_dict[key_A].device)
            for client in clients_this_round:
                lora_A_local = global_dict[key_A] + local_dict_list[client][key_A]
                lora_B_local = global_dict[key_B] + local_dict_list[client][key_B]
                w_local = lora_B_local @ lora_A_local
                avg_w_local += w_local * (sample_num_list[client] / sample_this_round)

            B_old = global_dict[key_B]
            A_old = global_dict[key_A]
            W_old = global_dict[key_B] @ global_dict[key_A]
            W_new = avg_w_local
            delta_w_target = W_new - W_old

            # Alternating logic based on the round index
            if round_idx % 2 == 0:
                print(f"  -> Round {round_idx} (even): Updating LoRA A")
                B_old_pinv = torch.linalg.pinv(B_old.float()).to(B_old.dtype)
                delta_A_proj = B_old_pinv @ delta_w_target
                # Sparsify the result with random mask
                sparse_delta_A = mask_input_with_mask_rate(input_tensor=delta_A_proj, mask_rate=fed_args.download_sparse_ratio, rescale=True, mask_strategy='random')
                sparse_delta_to_send[key_A] = sparse_delta_A
                sparse_delta_to_send[key_B] = torch.zeros_like(B_old)
            else:
                print(f"  -> Round {round_idx} (odd): Updating LoRA B")
                A_old_pinv = torch.linalg.pinv(A_old.float()).to(A_old.dtype)
                delta_B_proj = delta_w_target @ A_old_pinv
                # Sparsify the result with random mask
                sparse_delta_B = mask_input_with_mask_rate(input_tensor=delta_B_proj, mask_rate=fed_args.download_sparse_ratio, rescale=True, mask_strategy='random')
                sparse_delta_to_send[key_A] = torch.zeros_like(A_old)
                sparse_delta_to_send[key_B] = sparse_delta_B
            
        return sparse_delta_to_send
    
    else:   # Normal aggregation 
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
        return global_dict



def DARE(global_dict, local_dict, dare_p, mask_strategy='random'):
    delta_param = {}
    for key in global_dict.keys():
        delta_param[key] = local_dict[key] - global_dict[key]
    
    if dare_p == 0:
        return delta_param
    
    for param_name, param_value in delta_param.items():
        delta_param[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=dare_p, rescale=(mask_strategy=='random'), mask_strategy=mask_strategy)

    return delta_param

    
def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, rescale: bool, mask_strategy: str):
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"

    masked_input_tensor = input_tensor.clone()

    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)

    elif mask_strategy == "magnitude":
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)

    else:
        raise ValueError(f"unknown mask strategy: {mask_strategy}")

    if rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor
