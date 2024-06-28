import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
import os
import data_split

train_df, val_df, test_df =  data_split.split_dataset("../Dataset/autoirt.csv")

# Print dataset sizes
print(len(train_df), len(test_df), len(val_df))

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode data
train_encodings = tokenizer(list(train_df['element']), truncation=True, padding=True)
val_encodings = tokenizer(list(val_df['element']), truncation=True, padding=True)

# Define dataset class
class DefineDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset instances
train_dataset = DefineDataset(train_encodings, list(train_df['label']))
val_dataset = DefineDataset(val_encodings, list(val_df['label']))

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./BERT_training_output',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=8,
    warmup_steps=2000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    seed=120,
    optim="adamw_torch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    dataloader_num_workers=1,
)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits

file_labels = {
    'describe': 0,
    'expected': 1,
    'reproduce': 2,
    'actual': 3,
    'environment': 4,
    'additional': 5
}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    y_pred = np.argmax(predictions, axis=1)
    y_true = labels

    # Calculate overall metrics
    overall_f1 = f1_score(y_true, y_pred, average='weighted')
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='weighted')
    overall_recall = recall_score(y_true, y_pred, average='weighted')

    return {
        'accuracy': overall_accuracy,  # Return accuracy as evaluation metric
    }

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # Preprocess evaluation metrics
)

# Train model
trainer.train()
