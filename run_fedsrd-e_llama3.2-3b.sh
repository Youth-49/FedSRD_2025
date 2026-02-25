max_steps=10
num_rounds=100
batch_size=8
gradient_accumulation_steps=4
seq_length=512
num_clients=4
sample_clients=4
lora_r=64
lora_alpha=128   # twice of lora_r
lr=5e-5
download_sparse_ratio=0.8

dataset_list="sahil2801/CodeAlpaca-20k medalpaca/medical_meadow_medqa gbharti/finance-alpaca my_math1"
dataset_sample_list='-1 -1 -1 -1'
model_name_or_path="meta-llama/Llama-3.2-3B"
output_dir=./output

fed_alg="fedsrd-e"

python main_fedsrd.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_list $dataset_list \
 --dataset_sample_list $dataset_sample_list \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --download_sparse_ratio $download_sparse_ratio \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --peft_lora_target_modules all-linear \
 --simple_aggr \
 --use_peft \
 --output_dir $output_dir \
 --template "alpaca" \
