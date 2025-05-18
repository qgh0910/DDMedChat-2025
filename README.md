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
```bash
llamafactory-cli webui
```
## 5. Performance Evaluation
Use dragin_evaluation.py to evaluate the model's performance. Ensure that the correct dataset path and the ElasticSearch index name created by main.py are specified.
```bash
python dragin_evaluation.py --beir_corpus_path "./data/test_data" --index_name "model_index"
```
### Performance of Medical Large Language Models Fine-tuned with DoRA and RAG Models Based on DRAGIN on the Medical Dialogue Dataset
| Dataset           | Model                   | BLEU-4  | ROUGE-1  | ROUGE-L  |
|-------------------|--------------------------|--------:|---------:|---------:|
| Gastroenterology  | Qwen2-7b+ DRAGIN         | 18.8793 | 39.7877  | 40.3125  |
|                   | ChatGLM3-6b + DRAGIN     | 19.4113 | 40.0444  | 41.0543  |
|                   | Qwen2-7b+w/o RAG         | 9.2875  | 11.3856  | 18.7722  |
|                   | ChatGLM3-6b+w/o RAG      | 9.2856  | 11.5577  | 18.8668  |
|                   | ChatGLM3-6b**            | 12.1644 | 25.3406  | 26.6342  |
|                   | DDMedChat*               | 19.5617 | 39.6500  | 40.0214  |
|                   | DDMedChat**              | 19.8200 | 40.5072  | 40.4590  |
|                   | DDMedChat***             | 19.6525 | 40.9240  | 40.7358  |
| Internal Medicine | Qwen2-7b+ DRAGIN         | 17.8208 | 37.4655  | 38.2362  |
|                   | ChatGLM3-6b + DRAGIN     | 18.5057 | 38.1342  | 39.6373  |
|                   | Qwen2-7b+w/o RAG         | 9.2705  | 10.3908  | 19.9836  |
|                   | ChatGLM3-6b+w/o RAG      | 9.2378  | 11.3518  | 19.0349  |
|                   | ChatGLM3-6b**            | 12.2313 | 25.5939  | 26.7940  |
|                   | DDMedChat*               | 18.5593 | 37.6984  | 40.3105  |
|                   | DDMedChat**              | 18.6449 | 38.5966  | 39.8941  |
|                   | DDMedChat***             | 18.9236 | 39.4453  | 39.6580  |
| Otolaryngology    | Qwen2-7b+ DRAGIN         | 18.0640 | 38.4677  | 38.3220  |
|                   | ChatGLM3-6b + DRAGIN     | 18.5175 | 37.8565  | 38.5139  |
|                   | Qwen2-7b+w/o RAG         | 9.4987  | 9.7956   | 17.4727  |
|                   | ChatGLM3-6b+w/o RAG      | 9.0396  | 10.8386  | 18.5301  |
|                   | ChatGLM3-6b**            | 12.1089 | 25.3123  | 26.4456  |
|                   | DDMedChat*               | 19.3825 | 40.1850  | 41.0837  |
|                   | DDMedChat**              | 19.1443 | 41.3782  | 42.9685  |
|                   | DDMedChat***             | 19.3573 | 41.1621  | 41.7628  |
| Pediatrics        | Qwen2-7b+ DRAGIN         | 20.8368 | 42.2496  | 42.1096  |
|                   | ChatGLM3-6b + DRAGIN     | 20.0915 | 40.9212  | 41.5272  |
|                   | Qwen2-7b+w/o RAG         | 10.0548 | 13.1980  | 17.8556  |
|                   | ChatGLM3-6b+w/o RAG      | 9.7076  | 12.0465  | 19.7235  |
|                   | ChatGLM3-6b**            | 12.3510 | 25.5654  | 26.9745  |
|                   | DDMedChat*               | 19.5821 | 42.5326  | 43.3001  |
|                   | DDMedChat**              | 19.8082 | 42.5738  | 45.1504  |
|                   | DDMedChat***             | 19.7138 | 43.2985  | 44.0215  |

```bash
llamafactory-cli webui
```

