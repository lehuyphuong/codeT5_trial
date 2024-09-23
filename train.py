import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset, load_metric
from sklearn.model_selection import train_test_split
import pandas as pd

import os
os.environ["WANDB_DISABLED"] = "true"

# Load the model and tokenizer from the local directory
model_dir = "./codet5-base"  # Path to the downloaded model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Load your dataset (C code as input, JS code as target) from a JSON file
dataset = load_dataset("json", data_files={
                       "data": "D:/2024/Projects/C2F/Appoarch_1/CodeT5/Analysis_room/data/data.json"})["data"]

valid_dataset = [entry for entry in dataset if entry['input']
                 is not None and entry['target'] is not None]

# Split the dataset into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(
    valid_dataset, test_size=0.2, random_state=42)

# Convert lists to Pandas DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Convert to Hugging Face `Dataset` format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Preprocess the dataset for tokenization


def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]

    # Tokenize the input sequence
    model_inputs = tokenizer(inputs, max_length=512,
                             truncation=True, padding=True)

    # Tokenize the target sequence
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512,
                           truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply preprocessing to the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Output directory for the fine-tuned model
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,  # Initial learning rate
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    weight_decay=0.01,  # Weight decay
    save_total_limit=3,  # Limit the total number of saved checkpoints
    num_train_epochs=3,  # Number of training epochs
    predict_with_generate=True,  # Enables the model to generate sequences
    logging_dir='./logs',  # Directory for logging
)

# Load evaluation metric (BLEU score for sequence generation tasks)
metric = load_metric("sacrebleu")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    labels = [[label] for label in tokenizer.batch_decode(
        labels, skip_special_tokens=True)]

    # Compute BLEU score using sacrebleu
    bleu = metric.compute(predictions=decoded_preds, references=labels)
    return {"bleu": bleu["score"]}


# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model after training
model.save_pretrained(
    "D:/2024/Projects/C2F/Appoarch_1/CodeT5/Analysis_room/CP/fine-tuned-codeT5")
tokenizer.save_pretrained(
    "D:/2024/Projects/C2F/Appoarch_1/CodeT5/Analysis_room/CP/fine-tuned-codeT5")

print("Training completed and model saved.")
