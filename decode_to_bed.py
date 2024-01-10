#!/usr/bin/env python

import argparse

import h5py
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--targets_file", type=str, help="Path to HDF5 file")
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-large-t2t")
class_lables = np.array(["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic region"])


with h5py.File(args.targets_file, "r") as file:
    for transcript in list(file["records"].keys()):
        for sample in list(file["records"][transcript].keys()):
            token_ids_record = np.array(
                file["records"][transcript][sample]["token_ids"]
            )
            chars = tokenizer.convert_ids_to_tokens(token_ids_record)

            sample_start = np.array(
                file["records"][transcript][sample]["coordinates"][0]
            )
            sample_end = np.array(
                file["records"][transcript][sample]["coordinates"][-1]
            )

            classes = np.array(file["records"][transcript][sample]["classes"])

            chr = str(file["records"][transcript][sample]["info"][0]).split("'")[1]
            gene = str(file["records"][transcript][sample]["info"][1]).split("'")[1]
            trans = str(file["records"][transcript][sample]["info"][2]).split("'")[1]
            trans_class = str(file["records"][transcript][sample]["info"][3]).split(
                "'"
            )[1]
            strand = str(file["records"][transcript][sample]["info"][4]).split("'")[1]

            start = int(sample_start)
            for i, seq in enumerate(chars[1:-1]):
                column = np.array(classes[:, i+1])
                column_mask = (column > 0)

                print(
                    f"{chr}\t{start}\t{start + len(seq)}\t{class_lables[column_mask]};gene={gene};transcript={trans};class={trans_class};strand={strand};seq={seq}"
                )
                start += len(seq)
