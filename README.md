# FedSRD: Sparsify-Reconstruct-Decompose for Communication-Efficient Federated LLM Fine-Tuning

### Abstract

The current paradigm of training large language models (LLMs) on public available Web data is becoming unsustainable as high-quality data sources in specialized domains near exhaustion. Federated Learning (FL) emerges as a practical solution for the next generation of AI on a decentralized Web, enabling privacy-preserving collaborative fine-tuning on decentralized private data. While Low-Rank Adaptation (LoRA) is standard for efficient fine-tuning, its federated application faces a critical bottleneck: communication overhead under heterogeneous network conditions. Structural redundancy in LoRA parameters increases communication costs and causes aggregation conflicts. To address this, we propose FedSRD, a Sparsify-Reconstruct-Decompose framework for communication-efficient federated LLM fine-tuning. We introduce importance-aware sparsification to reduce the upload parameter count while preserving the structural integrity of LoRA updates. The server aggregates updates in full-rank space to mitigate conflicts, then decomposes the global update into a sparse low-rank format for broadcast, ensuring a symmetrically efficient cycle. We also propose an efficient variant, FedSRD-e, to reduce computational overhead. Experiments on 10 benchmarks show our framework significantly reduces communication costs by up to 90% while improving performance on heterogeneous client data.

---

### Overview
FedSRD (Sparsify‑Reconstruct‑Decompose) targets communication overhead in federated LoRA fine‑tuning (collaborative learning from heterogeneous domains):
- Client side: importance/structure‑aware sparsification of LoRA updates to reduce upload size.
- Server side: reconstruct and aggregate updates in full‑rank space, then decompose into sparse low‑rank updates for broadcast.
- FedSRD‑e: a lightweight variant that skips the SVD reconstruction step to reduce compute.

Beyond the pipeline itself, FedSRD offers three practical advantages:
- Communication‑efficient: aggressive yet structure‑preserving update compression substantially reduces communication cost.
- Effective cross‑domain knowledge merging: full‑rank reconstruction and aggregation help combine client updates from heterogeneous domains more effectively.
- Stable OOD knowledge retention: federated specialization preserves broader out‑of‑domain capabilities more reliably while adapting to domain‑specific data.

### Setup
1) Install dependencies
```
pip install -r requirements.txt
```
2) (Optional) set HuggingFace cache path  
Edit `hf_path_config.py`:
```
HF_CACHE_DIR='/path/to/your/hf_cache'
```

### Quick Start
#### FedSRD (Llama‑3.2‑3B)
```
bash run_fedsrd_llama3.2-3b.sh
```

#### FedSRD‑E (Llama‑3.2‑3B)
```
bash run_fedsrd-e_llama3.2-3b.sh
```

#### Qwen2‑7B
```
bash run_fedsrd_qwen2-7b.sh
bash run_fedsrd-e_qwen2-7b.sh
```

### Outputs
Run outputs are saved to a timestamped directory and include:
- `args.json`
- `training_log.json`
- `checkpoint-round*`

### Evaluation Benchmarks

| Category | Domain | Benchmark (Instance‑Shot) |
| --- | --- | --- |
| In‑domain | Code | HumanEval (0-shot); Sanitized MBPP (3-shot) |
| In‑domain | Medical | MedQA (1-shot); MedMCQA (1-shot) |
| In‑domain | Finance | FinEval (0-shot); FinanceIQ (0-shot) |
| In‑domain | Math | GSM8K (0-shot); MATH (0-shot) |
| Out‑of‑domain | General | AGIEval (0-shot) |
| Out‑of‑domain | Law | LawBench (1-shot) |

### Citation

```
@inproceedings{10.1145/3774904.3792144,
author = {Yan, Guochen and Xie, Luyuan and Shen, Qingni and Fang, Yuejian and Wu, Zhonghai},
title = {FedSRD: Sparsify-Reconstruct-Decompose for Communication-Efficient Federated Large Language Models Fine-Tuning},
year = {2026},
isbn = {9798400723070},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3774904.3792144},
doi = {10.1145/3774904.3792144},
abstract = {The current paradigm of training large language models (LLMs) on public available Web data is becoming unsustainable as high-quality data sources in specialized domains near exhaustion. Federated Learning (FL) emerges as a practical solution for the next generation of AI on a decentralized Web, enabling privacy-preserving collaborative fine-tuning on decentralized private data. While Low-Rank Adaptation (LoRA) is standard for efficient fine-tuning, its federated application faces a critical bottleneck: communication overhead under heterogeneous network conditions. Structural redundancy in LoRA parameters increases communication costs and causes aggregation conflicts. To address this, we propose FedSRD, a Sparsify-Reconstruct-Decompose framework for communication-efficient federated LLM fine-tuning. We introduce importance-aware sparsification to reduce the upload parameter count while preserving the structural integrity of LoRA updates. The server aggregates updates in full-rank space to mitigate conflicts, then decomposes the global update into a sparse low-rank format for broadcast, ensuring a symmetrically efficient cycle. We also propose an efficient variant, FedSRD-e, to reduce computational overhead. Experiments on 10 benchmarks show our framework significantly reduces communication costs by up to 90\% while improving performance on heterogeneous client data.},
booktitle = {Proceedings of the ACM Web Conference 2026},
pages = {5087–5098},
numpages = {12},
keywords = {federated learning, large language models, communication, lora},
location = {United Arab Emirates},
series = {WWW '26}
}
```
