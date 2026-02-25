import datasets
from datasets import load_dataset, load_from_disk
import pandas as pd
import os
import hf_path_config

def get_dataset(dataset_name, local_data_dir=None):

    if dataset_name in ["openai/gsm8k"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, cache_dir = hf_path_config.HF_CACHE_DIR, split="train", name="main")
    elif dataset_name in ["DigitalLearningGmbH/MATH-lighteval"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, cache_dir = hf_path_config.HF_CACHE_DIR, split="train", name="default")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, cache_dir = hf_path_config.HF_CACHE_DIR, split="train_sft")
    elif dataset_name in ['my_math1', ]:
        dataset_path = os.path.join(hf_path_config.HF_CACHE_DIR, dataset_name)
        dataset = load_from_disk(dataset_path)
    else:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, cache_dir = hf_path_config.HF_CACHE_DIR, split="train")

    return dataset

def process_sft_dataset(dataset_name, dataset, dataset_sample):
    if dataset_name in ["sahil2801/CodeAlpaca-20k", "lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train", 'medalpaca/medical_meadow_medqa', 'Alignment-Lab-AI/Lawyer-Instruct']:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified alpaca format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified alpaca format.")
    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["DigitalLearningGmbH/MATH-lighteval"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['level', 'type'])
    elif dataset_name in ['openai/gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ['my_math1', ]:
        pass
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample > 0:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example
