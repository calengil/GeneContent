#!/usr/bin/env python

import argparse

import pandas as pd
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--tokenized_seqs", type=str, help="Path to TXT file")
args = parser.parse_args()

data = pd.read_csv(args.tokenized_seqs, sep="\t", header=None)
tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-large-t2t")


def split_fields(info_record):
    records = info_record[1:-1].split(", ")
    return records


def to_chars(info_record):
    records = tokenizer.convert_ids_to_tokens(info_record)
    return records


data["class_0"] = data[10].apply(split_fields)
data["class_1"] = data[11].apply(split_fields)
data["class_2"] = data[12].apply(split_fields)
data["class_3"] = data[13].apply(split_fields)
data["class_4"] = data[14].apply(split_fields)
data["class_5"] = data[15].apply(split_fields)

data["ids"] = data[7].apply(split_fields)
data["chars"] = data["ids"].apply(to_chars)

class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic region"]


def main(df):
    start = int(df[5])
    for i, seq in enumerate(df["chars"]):
        elem = []
        classes = [
            df["class_0"][i],
            df["class_1"][i],
            df["class_2"][i],
            df["class_3"][i],
            df["class_4"][i],
            df["class_5"][i],
        ]
        for ind, stat in enumerate(classes):
            if stat == "1":
                elem.append(class_lables[ind])
        if len(elem) == 1 and elem[0] == "intergenic region":
            print(f"{df[3]}\t{start}\t{start + len(seq)}\t{';'.join(map(str, elem))}")
        else:
            print(
                f"{df[3]}\t{start}\t{start + len(seq)}\t{';'.join(map(str, elem))};gene={df[0]};transcript={df[1]};class={df[2]};strand={df[4]};seq={seq}"
            )
        start += len(seq)


data.apply(main, axis=1)
