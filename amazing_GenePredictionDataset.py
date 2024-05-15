#!/usr/bin/env python
import h5py
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class GeneAnnotationDataset(Dataset):
    def __init__(
        self,
        targets_file,
        tokenizer="AIRI-Institute/gena-lm-bigbird-base-t2t",
        max_tokens=4096,
        shift=255,
        label_number=3,
    ):

        self.data = h5py.File(targets_file, "r")
        self.samples = list(self.data.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_tokens = max_tokens - 2
        self.shift = shift
        self.label_number = label_number

        N_samples = []
        for transcript in self.samples:
          N_samples_per_transcript = 0
          for fragment_start in range(0, len(np.array(self.data[transcript]["input_ids"])), self.max_tokens - shift):
            N_samples_per_transcript += 1 
            if (fragment_start + self.max_tokens) >= len(np.array(self.data[transcript]["input_ids"])):
              break
          N_samples.append(N_samples_per_transcript)

        self.samples_cumsum = np.cumsum(N_samples)

    def __len__(self):
        return self.samples_cumsum[-1] - 1

    def __getitem__(self, idx):
        assert idx < self.samples_cumsum[-1]

        transcript_id = np.searchsorted(self.samples_cumsum, idx, "right")

        if transcript_id == 0:
          sample_id = idx
        else:
          sample_id = idx - self.samples_cumsum[transcript_id - 1]

        start = sample_id * (self.max_tokens - self.shift)
        end = start + self.max_tokens
        sample_name = "transcript_" + str(transcript_id)
        sample_size = len(np.array(self.data[sample_name]["input_ids"][start:end]))
        n_pads = self.max_tokens - sample_size
        
        if sample_size == self.max_tokens:
            input_ids = np.concatenate([[self.tokenizer.convert_tokens_to_ids("[CLS]")], 
                                        np.array(self.data[sample_name]["input_ids"][start:end]),
                                        [self.tokenizer.convert_tokens_to_ids("[SEP]")],])
            token_type_ids = np.concatenate([np.array([0]),
                                             np.array(self.data[sample_name]["token_type_ids"][start:end]),
                                             np.array([0])])
            attention_mask = np.concatenate([np.array([1]),
                                             np.array(self.data[sample_name]["attention_mask"][start:end]),
                                             np.array([1])])
            labels = np.concatenate([np.array([[-100]]*self.label_number),
                                     np.array(self.data[sample_name]["labels"][:, start:end]),
                                     np.array([[-100]]*self.label_number)], axis=1)
         
            coordinates = np.array(self.data[sample_name]["coordinates"][0] + start)
            info = np.array(self.data[sample_name]["info"])


        elif sample_size < self.max_tokens:
            input_ids = np.concatenate([[self.tokenizer.convert_tokens_to_ids("[CLS]")], 
                                        np.array(self.data[sample_name]["input_ids"][start:end]),
                                        [self.tokenizer.convert_tokens_to_ids("[PAD]")] * n_pads,
                                        [self.tokenizer.convert_tokens_to_ids("[SEP]")],])
            token_type_ids = np.concatenate([np.array([0]),
                                             np.array(self.data[sample_name]["token_type_ids"][start:end]),
                                             np.array([0] * n_pads),
                                             np.array([0])])
            attention_mask = np.concatenate([np.array([1]),
                                             np.array(self.data[sample_name]["attention_mask"][start:end]),
                                             np.array([0] * n_pads),
                                             np.array([1])])
            labels = np.concatenate([np.array([[-100]]*self.label_number),
                                     np.array(self.data[sample_name]["labels"][:, start:end]),
                                     np.array([[0]* n_pads]*self.label_number),
                                     np.array([[-100]]*self.label_number)], axis=1)
            coordinates = np.array(self.data[sample_name]["coordinates"][0] + start)
            info = np.array(self.data[sample_name]["info"])




        #labels = labels.T
        labels = labels.astype(np.float32)
        assert (
            len(input_ids) == self.max_tokens + 2
            and len(token_type_ids) == self.max_tokens + 2
            and len(attention_mask) == self.max_tokens + 2
        )

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels[[1,2,5]],
            "labels_mask": labels.min(axis=1) != -100,
            "coordinates": coordinates,
            "info": info,
        }

    def close(self):
        self.data.close()
