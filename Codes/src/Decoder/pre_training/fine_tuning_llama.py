import random
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json
import torch

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

file_path = "/data1/traffic_datasets/extracted_labels.json"

with open(file_path, "r") as f:
    data = json.load(f)

labels = data.get("labels")

# print(labels)
# print(len(labels))

data = {"labels": labels}
dataset = Dataset.from_dict(data)

model_name = "huggyllama/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")

def preprocess_function(examples):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(examples["labels"], padding=True, truncation=True)
    print(f"Original input shape: {len(examples['labels'])}, Tokenized input shape: {len(inputs['input_ids'])}")

    labels = inputs["input_ids"]

    print(f"Example input IDs: {inputs['input_ids'][:5]}")
    print(f"Example labels: {labels[:5]}")
    return {"input_ids": inputs["input_ids"], "labels": labels}

encoded_dataset = dataset.map(preprocess_function, batched=True)

# training paras
training_args = TrainingArguments(
    learning_rate=1e-5,
    output_dir="/data1/libin/llama-7b/results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    logging_dir="/data1/libin/llama-7b/logs",
    save_steps=2000,
    warmup_steps=500,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset
)

trainer.train()
