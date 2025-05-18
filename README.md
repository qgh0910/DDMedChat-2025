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
### Performance of Medical Large Language Models Fine-tuned with DoRA and RAG Models Based on DRAGIN on the Medical Dialogue Dataset
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Model</th>
      <th>BLEU-4</th>
      <th>ROUGE-1</th>
      <th>ROUGE-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8">Gastroenterology</td>
      <td>Qwen2-7b+ DRAGIN</td><td>18.8793</td><td>39.7877</td><td>40.3125</td>
    </tr>
    <tr><td>ChatGLM3-6b + DRAGIN</td><td>19.4113</td><td>40.0444</td><td>41.0543</td></tr>
    <tr><td>Qwen2-7b+w/o RAG</td><td>9.2875</td><td>11.3856</td><td>18.7722</td></tr>
    <tr><td>ChatGLM3-6b+w/o RAG</td><td>9.2856</td><td>11.5577</td><td>18.8668</td></tr>
    <tr><td>ChatGLM3-6b**</td><td>12.1644</td><td>25.34055</td><td>26.6342</td></tr>
    <tr><td>DDMedChat*</td><td>19.5617</td><td>39.6500</td><td>40.0214</td></tr>
    <tr><td>DDMedChat**</td><td>19.8200</td><td>40.5072</td><td>40.4590</td></tr>
    <tr><td>DDMedChat***</td><td>19.6525</td><td>40.9240</td><td>40.7358</td></tr>

    <tr>
      <td rowspan="8">Internal Medicine</td>
      <td>Qwen2-7b+ DRAGIN</td><td>17.8208</td><td>37.4655</td><td>38.2362</td>
    </tr>
    <tr><td>ChatGLM3-6b + DRAGIN</td><td>18.5057</td><td>38.1342</td><td>39.6373</td></tr>
    <tr><td>Qwen2-7b+w/o RAG</td><td>9.2705</td><td>10.3908</td><td>19.9836</td></tr>
    <tr><td>ChatGLM3-6b+w/o RAG</td><td>9.2378</td><td>11.3518</td><td>19.0349</td></tr>
    <tr><td>ChatGLM3-6b**</td><td>12.2313</td><td>25.5939</td><td>26.7940</td></tr>
    <tr><td>DDMedChat*</td><td>18.5593</td><td>37.6984</td><td>40.3105</td></tr>
    <tr><td>DDMedChat**</td><td>18.6449</td><td>38.5966</td><td>39.8941</td></tr>
    <tr><td>DDMedChat***</td><td>18.9236</td><td>39.4453</td><td>39.6580</td></tr>

    <tr>
      <td rowspan="8">Otolaryngology</td>
      <td>Qwen2-7b+ DRAGIN</td><td>18.0640</td><td>38.4677</td><td>38.3220</td>
    </tr>
    <tr><td>ChatGLM3-6b + DRAGIN</td><td>18.5175</td><td>37.8565</td><td>38.5139</td></tr>
    <tr><td>Qwen2-7b+w/o RAG</td><td>9.4987</td><td>9.7956</td><td>17.4727</td></tr>
    <tr><td>ChatGLM3-6b+w/o RAG</td><td>9.0396</td><td>10.8386</td><td>18.5301</td></tr>
    <tr><td>ChatGLM3-6b**</td><td>12.1089</td><td>25.3123</td><td>26.4456</td></tr>
    <tr><td>DDMedChat*</td><td>19.3825</td><td>40.1850</td><td>41.0837</td></tr>
    <tr><td>DDMedChat**</td><td>19.1443</td><td>41.3782</td><td>42.9685</td></tr>
    <tr><td>DDMedChat***</td><td>19.3573</td><td>41.1621</td><td>41.7628</td></tr>

    <tr>
      <td rowspan="8">Pediatrics</td>
      <td>Qwen2-7b+ DRAGIN</td><td>20.8368</td><td>42.2496</td><td>42.1096</td>
    </tr>
    <tr><td>ChatGLM3-6b + DRAGIN</td><td>20.0915</td><td>40.9212</td><td>41.5272</td></tr>
    <tr><td>Qwen2-7b+w/o RAG</td><td>10.0548</td><td>13.1980</td><td>17.8556</td></tr>
    <tr><td>ChatGLM3-6b+w/o RAG</td><td>9.7076</td><td>12.0465</td><td>19.7235</td></tr>
    <tr><td>ChatGLM3-6b**</td><td>12.3510</td><td>25.5654</td><td>26.9745</td></tr>
    <tr><td>DDMedChat*</td><td>19.5821</td><td>42.5326</td><td>43.3001</td></tr>
    <tr><td>DDMedChat**</td><td>19.8082</td><td>42.5738</td><td>45.1504</td></tr>
    <tr><td>DDMedChat***</td><td>19.7138</td><td>43.2985</td><td>44.0215</td></tr>
  </tbody>
</table>

```bash
llamafactory-cli webui
```

