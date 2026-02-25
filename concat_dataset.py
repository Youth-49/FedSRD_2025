import os
from datasets import load_dataset, concatenate_datasets, load_from_disk

cache_dir = ''
merged_dataset_path = ''

if os.path.exists(merged_dataset_path):
    merged_dataset = load_from_disk(merged_dataset_path)
else:
    gsm8k_train = load_dataset(
        "openai/gsm8k",
        name="main",
        split="train",
        cache_dir=cache_dir
    )
    gsm8k_train = gsm8k_train.rename_column("question", "instruction")
    gsm8k_train = gsm8k_train.rename_column("answer", "response")

    lighteval_train = load_dataset(
        "DigitalLearningGmbH/MATH-lighteval",
        name="default",
        split="train",
        cache_dir=cache_dir
    )
    lighteval_train = lighteval_train.rename_column("solution", "response")
    lighteval_train = lighteval_train.rename_column("problem", "instruction")
    lighteval_train = lighteval_train.remove_columns(['level', 'type'])

    merged_dataset = concatenate_datasets([gsm8k_train, lighteval_train])
    merged_dataset.save_to_disk(merged_dataset_path)


print(merged_dataset)
print(merged_dataset[0])