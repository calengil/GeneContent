#!/usr/bin/env python

import h5py
import numpy as np
from datetime import datetime
import importlib

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator


from modeling_bert import BertForTokenClassification
from GenePredictionDataset import GeneAnnotationDataset

from transformers.integrations import TensorBoardCallback

tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")
model = BertForTokenClassification.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t", 
                                                   num_labels=6,
                                                   problem_type = 'multi_label_classification')
train_d = GeneAnnotationDataset("/beegfs/data/hpcws/ws1/shadskii-gena/bio_data/chr9.hdf5")
valid_d = GeneAnnotationDataset("/beegfs/data/hpcws/ws1/shadskii-gena/bio_data/valid.hdf5")

data_collator = DefaultDataCollator()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = predictions[:, 1:predictions.shape[1]-1]
    labels = labels[:, 1:labels.shape[1]-1]
    
    TP = np.sum((labels == 1) & (predictions > 0.5))
    TN = np.sum((labels == 0) & (predictions <= 0.5))
    FP = np.sum((labels == 0) & (predictions > 0.5))
    FN = np.sum((labels == 1) & (predictions <= 0.5))

    return {"accuracy": (TP + TN) / (TP + TN + FP + FN),
           "recall": TP / (TP + FN),
           "precision": TP / (TP + FP),
           "f1_score": 2 * (TP / (TP + FP) * TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN))}
    

training_args = TrainingArguments(
    output_dir="/beegfs/data/hpcws/ws1/shadskii-gena/finetuning",
    learning_rate=2e-05,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
    optim="adamw_torch",
    weight_decay=0.0,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_d,
    eval_dataset=valid_d,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TensorBoardCallback],
)

trainer.train()

train_d.close()
valid_d.close()
