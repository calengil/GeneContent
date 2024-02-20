#!/usr/bin/env python

import h5py
import numpy as np
import importlib

import torch

torch.cuda.empty_cache()

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator

from modeling_bert import BertForTokenClassification
from short_GenePredictionDataset import short_GeneAnnotationDataset
from short_val_GenePredictionDataset import short_val_GeneAnnotationDataset

from transformers.integrations import TensorBoardCallback

tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")

model = BertForTokenClassification.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t", 
                                                   num_labels=6,
                                                   problem_type = 'multi_label_classification')

train_d = short_GeneAnnotationDataset("/beegfs/data/hpcws/ws1/shadskii-gena/bio_data/new_chr9.hdf5")
valid_d = short_val_GeneAnnotationDataset("/beegfs/data/hpcws/ws1/shadskii-gena/bio_data/new_validation.hdf5")

data_collator = DefaultDataCollator()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = predictions[:, 1:predictions.shape[1]-1]
    labels = labels[:, 1:labels.shape[1]-1]
    
    classes = ["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic"]
    my_metrics = {}
    for col, name in enumerate(classes):
        TP = np.sum((labels[:,:,col] == 1) & (predictions[:,:,col] > 0.5))
        TN = np.sum((labels[:,:,col] == 0) & (predictions[:,:,col] <= 0.5))
        FP = np.sum((labels[:,:,col] == 0) & (predictions[:,:,col] > 0.5))
        FN = np.sum((labels[:,:,col] == 1) & (predictions[:,:,col] <= 0.5))
        my_metrics[f"recall_{name}"] = TP / (TP + FN)
        my_metrics[f"precision_{name}"] = TP / (TP + FP)
        my_metrics[f"accuracy_{name}"] = (TP + TN) / (TP + TN + FP + FN)  

    TP = np.sum((labels == 1) & (predictions > 0.5))
    TN = np.sum((labels == 0) & (predictions <= 0.5))
    FP = np.sum((labels == 0) & (predictions > 0.5))
    FN = np.sum((labels == 1) & (predictions <= 0.5))

    my_metrics["recall"] = TP / (TP + FN)
    my_metrics["precision"] = TP / (TP + FP)
    my_metrics["accuracy"] = (TP + TN) / (TP + TN + FP + FN)  

    return my_metrics    

training_args = TrainingArguments(
    output_dir="/beegfs/data/hpcws/ws1/shadskii-gena/finetuning",
    learning_rate=2e-05,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
    optim="adamw_torch",
    weight_decay=0.0,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    label_names=["labels"],
    dataloader_num_workers=16,
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
