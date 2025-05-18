# DDMedChat：DoRA Fine-Turning and DRAGIN RAG based Medical Chat  Framework with LLMs
## Introduction
DDMedChat is a framework specifically designed for Chinese medical dialogue, which enhances the training efficiency and accuracy of Chinese medical interaction tasks by integrating DoRA's weight-decomposed low-rank adaptation fine-tuning technique with DRAGIN's dynamic retrieval-augmented generation technology.

## Environmental Requirements
- Python 3.8+
- Elasticsearch 7.x+

## Install Dependencies
Install the required Python libraries:

```bash
pip install -r requirements.txt
```
## 1. Construct a dataset

This project utilizes the Huatuo-Lite, MedDialog-CN, and Kuake-IR datasets for training and evaluation. Please follow the steps below to prepare and process the data.

Download the data
Go to https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite/tree/main, https://drive.google.com/drive/folders/1r09_i8nJ9c1nliXVGXwSqRYqklcHd9e2, and https://tianchi.aliyun.com/dataset/95414 to download the datasets.

Run the following commands for data preprocessing to ensure that the data can be correctly read and processed by the model.

```bash
python ./data/preprocess_data.py --input_path "./data/raw_data" --output_path "./data/processed_data"
```
### Dataset Types and Quantity Analysis
| Dataset       | Data Type              | Data Quantity   |
|---------------|------------------------|-----------------|
| Kuake-IR      | Medical Text Records   | 127,500         |
| Huatuo-26M    | Medical Dialogue Pairs | 26,000,000      |
| Huatuo-Lite   | Medical Dialogue Pairs | 180,000         |
| MedDialog-CN  | Medical Dialogue Pairs | 34,000,000      |

## 2. Config
```
{
    "model_name_or_path": "ChatGLM3-6b", 
    "method": "attn_entropy",
    "dataset": "huatuo-lite",
    "data_path": "../data/raw_data",
    "generate_max_length": 64,
    "query_formulation": "real_words",
    "retrieve_keep_top_k": 40,
    "output_dir": "../result/ChatGLM3-6b",
    "retriever": "BM25",
    "retrieve_topk": 3,
    "hallucination_threshold": 1.2,
    "fewshot": 6,
    "sample": 1000,
    "shuffle": false,
    "check_real_words": true,
    "es_index_name": "34051_wiki",
    "use_counter": true
}
```
## 3. Run

After the data preparation is completed, run the following commands in the directory.

```bash
python main.py -c path_to_config_file
```
## 4 Fine-tuning
This project utilizes LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) for model fine-tuning.
### step1： Install
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
### step2： Data Preparation
Refer to sft/data/README_zh.md to prepare fine-tuning data. When using a custom dataset, please update the sft/data/dataset_info.json file.
### step3：Fine-tuning and Evaluation (with Visual UI)
## 5. Performance Evaluation
Use dragin_evaluation.py to evaluate the model's performance. Ensure that the correct dataset path and the ElasticSearch index name created by main.py are specified.
```bash
python dragin_evaluation.py --beir_corpus_path "./data/test_data" --index_name "model_index"
```


```bash
llamafactory-cli webui
```

