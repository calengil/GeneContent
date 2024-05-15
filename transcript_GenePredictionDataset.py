#!/usr/bin/env python
import h5py
import numpy as np
from torch.utils.data import Dataset


class short_GeneAnnotationDataset(Dataset):
    def __init__(
        self,
        targets_file,
        tokenizer="AIRI-Institute/gena-lm-bert-large-t2t",
    ):
        self.data = h5py.File(targets_file, "r")

        self.samples = len(list(self.data.keys()))
        self.tokenizer = tokenizer

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        assert idx < self.samples

        sample_name = "transcript_" + str(idx)

        input_ids = np.array(
            self.data[sample_name]["input_ids"]
        )
        token_type_ids = np.array(
            self.data[sample_name]["token_type_ids"]
        )
        attention_mask = np.array(
            self.data[sample_name]["attention_mask"]
        )
        labels = np.array(
            self.data[sample_name]["labels"]
        )
        labels = labels.T
        labels = labels.astype(np.float32)
            
            
        assert (
            len(input_ids) == len(token_type_ids)
            and len(token_type_ids) == len(attention_mask)
        )

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "labels_mask": labels.min(axis=1) != -100
        }

    def close(self):
        self.data.close()
