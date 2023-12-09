#!/usr/bin/env python
import h5py
import numpy as np
from torch.utils.data import Dataset


class GeneAnnotationDataset(Dataset):
    def __init__(
        self,
        targets_file,
        tokenizer="AIRI-Institute/gena-lm-bert-large-t2t",
        max_seq_len=512,
    ):
        with h5py.File(targets_file, "r") as file:
            N_samples_per_gene = []
            for num, trans in enumerate(file["records"]):
                N_samples_per_gene.append(len(file["records"][trans]))

        self.saples_per_gene = N_samples_per_gene
        self.samples_cumsum = np.cumsum(N_samples_per_gene)
        self.data = targets_file
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.samples_cumsum[-1]

    def __getitem__(self, idx):
        assert idx < self.samples_cumsum[-1]

        gene_id = np.searchsorted(self.samples_cumsum, idx, "right")
        if gene_id == 0:
            sample_id = idx
        else:
            sample_id = idx - self.samples_cumsum[gene_id - 1]

        assert sample_id <= self.saples_per_gene[gene_id]

        sample_name = "sample_" + str(sample_id)
        with h5py.File(self.data, "r") as file:
            input_ids = np.array(
                file["records"][list(file["records"].keys())[gene_id]][sample_name][
                    "token_ids"
                ]
            )
            token_type_ids = np.array(
                file["records"][list(file["records"].keys())[gene_id]][sample_name][
                    "token_types"
                ]
            )
            attention_mask = np.array(
                file["records"][list(file["records"].keys())[gene_id]][sample_name][
                    "attention_mask"
                ]
            )
            labels = np.array(
                file["records"][list(file["records"].keys())[gene_id]][sample_name][
                    "classes"
                ]
            )
            assert (
                len(input_ids) == self.max_seq_len
                and len(token_type_ids) == self.max_seq_len
                and len(attention_mask) == self.max_seq_len
            )

            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
