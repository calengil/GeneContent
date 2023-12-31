#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import h5py
import numpy as np
import pandas as pd
import pysam
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--chrs", type=str, default="all", help="Select the chromosomes")
parser.add_argument("--gff", type=str, help="Path to GFF file")
parser.add_argument("--fasta", type=str, help="Path to FASTA file")
parser.add_argument("--fai", type=str, help="Path to FAI file")
parser.add_argument("--inp", type=int, default=512, help="number of input tokens")
parser.add_argument("--output", type=str, help="Path to output HDF5 file")
parser.add_argument("--shift", type=int, default=256, help="shift")
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default="AIRI-Institute/gena-lm-bert-large-t2t",
    help="tokenizer",
)
parser.add_argument("--radius", type=int, default=64, help="radius")
args = parser.parse_args()

col_names = [
    "seqid",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
]
data = pd.read_csv(
    args.gff, sep="\t", names=col_names, header=None, comment="#"
)
ref = pysam.Fastafile(args.fasta)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
chromsizes = pd.read_csv(args.fai, sep="\t", header=None)
if args.chrs != "all":
    selected_chrs = pd.read_csv(args.chrs, header=None)
    selected_chrs = selected_chrs[0].tolist()
    data = data[data["seqid"].isin(selected_chrs)]
    data.reset_index(drop=True, inplace=True)

trans_class_dict = {
    "lnc_RNA": "lnc_RNA",
    "mRNA": "mRNA",
}


def split_info_fields(info_record):
    records = info_record.split(";")
    result = {}
    for record in records:
        record_data = record.split("=")
        result[record_data[0]] = record_data[1]
    return result


new_cols = pd.DataFrame.from_records(data["attributes"].apply(split_info_fields))
data = pd.concat((data, new_cols), axis=1)

data_grouped = data.groupby("Parent")

data_lnc_RNA = data["type"] == "lnc_RNA"
data_mRNA = data["type"] == "mRNA"
transcript_name_mask = data_mRNA | data_lnc_RNA

valid_transcripts = data[transcript_name_mask]["ID"].unique()
np.set_printoptions(threshold=np.inf)

assert (
    len(
        np.intersect1d(
            data.query("Parent in @valid_transcripts")["ID"].unique(),
            data["Parent"].dropna().unique(),
        )
    )
    == 0
)
transcripts_index = data[transcript_name_mask].set_index("ID")


# Special tokens
def get_service_token_encodings(tokenizer):
    SEP_token_id = tokenizer.sep_token_id
    CLS_token_id = tokenizer.cls_token_id

    CLS_encoding = {
        "input_ids": np.array(CLS_token_id).reshape(-1, 1),
        "token_type_ids": np.array([0]).reshape(-1, 1),
        "attention_mask": np.array([1]).reshape(-1, 1),
    }

    SEP_encoding = {
        "input_ids": np.array(SEP_token_id).reshape(-1, 1),
        "token_type_ids": np.array([0]).reshape(-1, 1),
        "attention_mask": np.array([1]).reshape(-1, 1),
    }

    return {"CLS": CLS_encoding, "SEP": SEP_encoding}


# Add special tokens
def select_tokens(
    tokens_for_choosing,
    first_selected_token_index,
    last_selected_token_index,
    special_tokens,
):
    token_ids = np.array(
        transcript_part_tokens["input_ids"][
            first_selected_token_index : last_selected_token_index + 1
        ]
    )
    token_types = np.array(
        transcript_part_tokens["token_type_ids"][
            first_selected_token_index : last_selected_token_index + 1
        ]
    )
    attention_mask = np.array(
        transcript_part_tokens["attention_mask"][
            first_selected_token_index : last_selected_token_index + 1
        ]
    )

    token_ids = np.insert(token_ids, 0, special_tokens["CLS"]["input_ids"])
    token_types = np.insert(token_types, 0, special_tokens["CLS"]["token_type_ids"])
    attention_mask = np.insert(
        attention_mask, 0, special_tokens["CLS"]["attention_mask"]
    )

    token_ids = np.append(token_ids, special_tokens["SEP"]["input_ids"])
    token_types = np.append(token_types, special_tokens["SEP"]["token_type_ids"])
    attention_mask = np.append(attention_mask, special_tokens["SEP"]["attention_mask"])

    return token_ids, token_types, attention_mask


# Get part of DNA sequence
def tokenize_sequence(
    trans_name, start_for_tokenize, tokens_count, chr_name, chr_info, reference, radius
):
    start_seq = start_for_tokenize - radius * tokens_count
    end_seq = start_for_tokenize + radius * tokens_count

    chr_length = chr_info[chr_info[0] == chr_name][1].values[0]

    if start_seq < 0:
        start_seq = 0
    elif end_seq > chr_length:
        end_seq = chr_length

    assert (
        end_seq <= chr_length and start_seq >= 0
    ), "DNA beyond the boundaries of the chromosome"

    DNA_seq = reference.fetch(reference=chr_name, start=start_seq, end=end_seq)
    DNA_seq = DNA_seq.upper()

    # Tokenize part of DNA sequence
    tokens = tokenizer.tokenize(DNA_seq)
    token_to_chars = tokenizer.encode_plus(
        DNA_seq, add_special_tokens=False, return_offsets_mapping=True
    )
    mapping_tokens = token_to_chars["offset_mapping"]
    while len(mapping_tokens) < args.inp - 2:
        if start_seq > 0:
            if start_seq - 500 >= 0:
                start_seq = start_seq - 500
            else:
                start_seq = 0
        if end_seq < chr_length:
            if end_seq + 500 <= chr_length:
                end_seq = end_seq + 500
            else:
                end_seq = chr_length

        DNA_seq = reference.fetch(reference=chr_name, start=start_seq, end=end_seq)
        DNA_seq = DNA_seq.upper()

        token_to_chars = tokenizer.encode_plus(
            DNA_seq, add_special_tokens=False, return_offsets_mapping=True
        )
        mapping_tokens = token_to_chars["offset_mapping"]
        if start_seq == 0 and end_seq == chr_length:
            break
    # Find start for tokenize for coordinates of tokens
    center = start_for_tokenize - start_seq
    return token_to_chars, center


def select_part(token_to_chars, center, inp):
    # Select part of sequence
    mapping_tokens = token_to_chars["offset_mapping"]

    assert len(mapping_tokens) == len(set(mapping_tokens))
    assert (
        mapping_tokens[0][0] <= center <= mapping_tokens[-1][-1]
    ), "no center in tokens"

    for interval in mapping_tokens:
        if interval[0] <= center <= interval[1]:
            central_token = interval

    tokens_for_classing = [central_token]
    count_of_tokens = 1
    right_len = central_token[1] - center
    left_len = center - central_token[0]

    while count_of_tokens < inp - 2:
        if (
            tokens_for_classing[0] == mapping_tokens[0]
            and tokens_for_classing[-1] == mapping_tokens[-1]
        ):
            break
        else:
            if right_len >= left_len:
                if tokens_for_classing[0] != mapping_tokens[0]:
                    tokens_for_classing.insert(
                        0,
                        mapping_tokens[
                            mapping_tokens.index(tokens_for_classing[0]) - 1
                        ],
                    )
                    left_len = center - tokens_for_classing[0][0]
                    count_of_tokens += 1
                else:
                    tokens_for_classing.append(
                        mapping_tokens[
                            mapping_tokens.index(tokens_for_classing[-1]) + 1
                        ]
                    )
                    right_len = tokens_for_classing[-1][1] - center
                    count_of_tokens += 1
            elif right_len < left_len:
                if tokens_for_classing[-1] != mapping_tokens[-1]:
                    tokens_for_classing.append(
                        mapping_tokens[
                            mapping_tokens.index(tokens_for_classing[-1]) + 1
                        ]
                    )
                    right_len = tokens_for_classing[-1][1] - center
                    count_of_tokens += 1
                else:
                    tokens_for_classing.insert(
                        0,
                        mapping_tokens[
                            mapping_tokens.index(tokens_for_classing[0]) - 1
                        ],
                    )
                    left_len = center - tokens_for_classing[0][0]
                    count_of_tokens += 1

    for ind, tok in enumerate(tokens_for_classing):
        assert tok[1] > tok[0]
        if ind < len(tokens_for_classing) - 1:
            assert tokens_for_classing[ind][1] <= tokens_for_classing[ind + 1][0]

    return tokens_for_classing


def search_exons_end_cds(trans_content, strand):
    # first and last exon - for search intron
    exon_cotent = trans_content[trans_content["type"] == "exon"]
    if exon_cotent.empty:
        first_exon_start = -1000
        last_exon_end = -1000
    else:
        if strand == "+":
            first_exon_start = exon_cotent["start"].values.min()
            last_exon_end = exon_cotent["end"].values.max()
        elif strand == "-":
            first_exon_start = exon_cotent["end"].values.max()
            last_exon_end = exon_cotent["start"].values.min()

    # first and last CDS - for search UTR
    cds_cotent = trans_content[trans_content["type"] == "CDS"]
    if cds_cotent.empty:
        first_cds_start = -1000
        last_cds_end = -1000
    else:
        if strand == "+":
            first_cds_start = cds_cotent["start"].values.min()
            last_cds_end = cds_cotent["end"].values.max()
        elif strand == "-":
            first_cds_start = cds_cotent["end"].values.max()
            last_cds_end = cds_cotent["start"].values.min()

    return first_exon_start, last_exon_end, first_cds_start, last_cds_end


# classification + strand
def classification_forward(
    tokens_for_classing,
    trans_content,
    first_exon_start,
    last_exon_end,
    first_cds_start,
    last_cds_end,
    start_for_tokenize,
    center,
):
    class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic"]
    classes = np.zeros(shape=(len(class_lables), len(tokens_for_classing)), dtype=int)
    final_token_class = []

    # prior classify tokens
    for i, tok in enumerate(tokens_for_classing):
        token_start = start_for_tokenize - center + tok[0]
        token_end = start_for_tokenize - center + tok[1]
        prior_token_class = []
        if token_start <= last_exon_end and token_end >= first_exon_start:
            if token_start < first_exon_start or token_end > last_exon_end:
                prior_token_class.append("intergenic")
            find_exon = (
                (trans_content["type"] == "exon")
                & (trans_content["start"] <= token_end)
                & (trans_content["end"] >= token_start)
            )
            if trans_content[find_exon].empty:
                prior_token_class.append("intron")
            else:
                prior_token_class.append("exon")
                if first_cds_start != -1000:
                    # Search UTR
                    if first_cds_start > first_exon_start:
                        if token_start < first_cds_start:
                            prior_token_class.append("5UTR")
                    if last_cds_end < last_exon_end:
                        if token_end > last_cds_end:
                            prior_token_class.append("3UTR")
                # Search introns
                on_left_of_exon = (
                    (trans_content["type"] == "exon")
                    & (trans_content["start"] > token_start)
                    & (trans_content["start"] <= token_end)
                    & (trans_content["start"] > first_exon_start)
                )
                if not trans_content[on_left_of_exon].empty:
                    prior_token_class.append("intron")
                on_right_of_exon = (
                    (trans_content["type"] == "exon")
                    & (trans_content["end"] >= token_start)
                    & (trans_content["end"] < token_end)
                    & (trans_content["end"] < last_exon_end)
                )
                if not trans_content[on_right_of_exon].empty:
                    prior_token_class.append("intron")
        else:
            prior_token_class.append("intergenic")
            # Search CDS
            find_cds = (
                (trans_content["type"] == "CDS")
                & (trans_content["start"] <= token_end)
                & (trans_content["end"] >= token_start)
            )
            if not trans_content[find_cds].empty:
                prior_token_class.append("CDS")
        final_token_class.append(prior_token_class)

    for tok_index, tok in enumerate(tokens_for_classing):
        for l_index, label in enumerate(class_lables):
            if label in final_token_class[tok_index]:
                classes[l_index, tok_index] = 1
    classes = np.array(classes)
    classes = np.insert(classes, 0, -100, axis=1)
    classes = np.insert(classes, classes.shape[1], -100, axis=1)
    return classes


# classification - strand
def classification_reverse(
    tokens_for_classing,
    trans_content,
    first_exon_start,
    last_exon_end,
    first_cds_start,
    last_cds_end,
    start_for_tokenize,
    center,
):
    class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic"]
    classes = np.zeros(shape=(len(class_lables), len(tokens_for_classing)), dtype=int)
    final_token_class = []

    # prior classify tokens
    for i, tok in enumerate(tokens_for_classing):
        token_start = start_for_tokenize - center + tok[0]
        token_end = start_for_tokenize - center + tok[1]
        prior_token_class = []
        if token_start <= first_exon_start and token_end >= last_exon_end:
            if token_start < last_exon_end or token_end > first_exon_start:
                prior_token_class.append("intergenic")
            find_exon = (
                (trans_content["type"] == "exon")
                & (trans_content["start"] <= token_end)
                & (trans_content["end"] >= token_start)
            )
            if trans_content[find_exon].empty:
                prior_token_class.append("intron")
            else:
                prior_token_class.append("exon")
                if first_cds_start != -1000:
                    # Search UTR
                    if first_cds_start < first_exon_start:
                        if token_end > first_cds_start:
                            prior_token_class.append("5UTR")
                    if last_cds_end > last_exon_end:
                        if token_start < last_cds_end:
                            prior_token_class.append("3UTR")
                # Search introns
                on_left_of_exon = (
                    (trans_content["type"] == "exon")
                    & (trans_content["start"] > token_start)
                    & (trans_content["start"] <= token_end)
                    & (trans_content["start"] > last_exon_end)
                )
                if not trans_content[on_left_of_exon].empty:
                    prior_token_class.append("intron")
                on_right_of_exon = (
                    (trans_content["type"] == "exon")
                    & (trans_content["end"] >= token_start)
                    & (trans_content["end"] < token_end)
                    & (trans_content["end"] < first_exon_start)
                )
                if not trans_content[on_right_of_exon].empty:
                    prior_token_class.append("intron")
        else:
            prior_token_class.append("intergenic")
            # Search CDS
            find_cds = (
                (trans_content["type"] == "CDS")
                & (trans_content["start"] <= token_end)
                & (trans_content["end"] >= token_start)
            )
            if not trans_content[find_cds].empty:
                prior_token_class.append("CDS")
        final_token_class.append(prior_token_class)

    for tok_index, tok in enumerate(tokens_for_classing):
        for l_index, label in enumerate(class_lables):
            if label in final_token_class[tok_index]:
                classes[l_index, tok_index] = 1
    classes = np.array(classes)
    classes = np.insert(classes, 0, -100, axis=1)
    classes = np.insert(classes, classes.shape[1], -100, axis=1)
    return classes


# Let's do it
with h5py.File(args.output, "w") as file:
    run_info = file.create_group("run_info")
    array_run_info = [
        "GFF=" + args.gff,
        "FASTA=" + args.fasta,
        "FAI=" + args.fai,
        "number of input tokens=" + str(args.inp),
        "output HDF5=" + args.output,
        "shift=" + str(args.shift),
        "tokenizer=" + args.tokenizer_name,
        "radius=" + str(args.radius),
    ]
    array_info = run_info.create_dataset("info", data=array_run_info)

    transcripts_records = file.create_group("records")
    for transcript in valid_transcripts:
        index_sample = 0
        group = transcripts_records.create_group(transcript)
        transcript_name = transcript
        transcript_content = data_grouped.get_group(transcript)
        transcript_info = transcripts_index.loc[transcript]
        gene = transcript_info["Parent"]
        transcript_strand = transcript_info["strand"]
        transcript_chr = transcript_info["seqid"]
        transcript_class = transcript_info["type"]
        if transcript_strand == "+":
            transcript_start = transcript_info["start"]
            transcript_end = transcript_info["end"]
        elif transcript_strand == "-":
            transcript_start = transcript_info["end"]
            transcript_end = transcript_info["start"]
        assert (
            transcript_strand == "+" or transcript_strand == "-"
        ), "Not identifited strand"

        (
            first_exon_start,
            last_exon_end,
            first_cds_start,
            last_cds_end,
        ) = search_exons_end_cds(transcript_content, transcript_strand)
        start_for_tokenize = transcript_start
        end_out = transcript_end
        start_out = transcript_end
        transcript_stats = [
            transcript_chr,
            gene,
            transcript_name,
            trans_class_dict[transcript_class],
            transcript_strand,
        ]

        if transcript_strand == "+":
            while end_out <= transcript_end:
                transcript_part_tokens, center = tokenize_sequence(
                    transcript_name,
                    start_for_tokenize,
                    args.inp,
                    transcript_chr,
                    chromsizes,
                    ref,
                    args.radius,
                )
                if len(transcript_part_tokens["offset_mapping"]) < args.inp - 2:
                    break
                assert len(transcript_part_tokens["offset_mapping"]) >= args.inp - 2
                selected_tokens_coor = select_part(
                    transcript_part_tokens, center, args.inp
                )
                classes = classification_forward(
                    selected_tokens_coor,
                    transcript_content,
                    first_exon_start,
                    last_exon_end,
                    first_cds_start,
                    last_cds_end,
                    start_for_tokenize,
                    center,
                )
                start_out = start_for_tokenize - center + selected_tokens_coor[0][0]
                end_out = start_for_tokenize - center + selected_tokens_coor[-1][1]
                first_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[0])
                last_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[-1])

                special_tokens = get_service_token_encodings(tokenizer)
                token_ids, token_types, attention_mask = select_tokens(
                    transcript_part_tokens,
                    first_selected_token_index,
                    last_selected_token_index,
                    special_tokens,
                )
                coordinates = [int(start_out), int(end_out)]

                subgrp = group.create_group(f"sample_{index_sample}")

                array_info = subgrp.create_dataset("info", data=transcript_stats)
                array_coordinates = subgrp.create_dataset(
                    "coordinates", data=coordinates
                )
                array_ids = subgrp.create_dataset("token_ids", data=token_ids)
                array_types = subgrp.create_dataset("token_types", data=token_types)
                array_mask = subgrp.create_dataset(
                    "attention_mask", data=attention_mask
                )
                array_classes = subgrp.create_dataset("classes", data=classes)

                index_sample += 1

                start_for_tokenize += args.shift

        elif transcript_strand == "-":
            while start_out >= transcript_end:
                transcript_part_tokens, center = tokenize_sequence(
                    transcript_name,
                    start_for_tokenize,
                    args.inp,
                    transcript_chr,
                    chromsizes,
                    ref,
                    args.radius,
                )
                if len(transcript_part_tokens["offset_mapping"]) < args.inp - 2:
                    break
                assert len(transcript_part_tokens["offset_mapping"]) >= args.inp - 2
                selected_tokens_coor = select_part(
                    transcript_part_tokens, center, args.inp
                )
                classes = classification_reverse(
                    selected_tokens_coor,
                    transcript_content,
                    first_exon_start,
                    last_exon_end,
                    first_cds_start,
                    last_cds_end,
                    start_for_tokenize,
                    center,
                )
                start_out = start_for_tokenize - center + selected_tokens_coor[0][0]
                end_out = start_for_tokenize - center + selected_tokens_coor[-1][1]
                first_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[0])
                last_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[-1])

                special_tokens = get_service_token_encodings(tokenizer)
                token_ids, token_types, attention_mask = select_tokens(
                    transcript_part_tokens,
                    first_selected_token_index,
                    last_selected_token_index,
                    special_tokens,
                )
                coordinates = [int(start_out), int(end_out)]

                subgrp = group.create_group(f"sample_{index_sample}")
                array_info = subgrp.create_dataset("info", data=transcript_stats)
                array_coordinates = subgrp.create_dataset(
                    "coordinates", data=coordinates
                )
                array_ids = subgrp.create_dataset("token_ids", data=token_ids)
                array_types = subgrp.create_dataset("token_types", data=token_types)
                array_mask = subgrp.create_dataset(
                    "attention_mask", data=attention_mask
                )
                array_classes = subgrp.create_dataset("classes", data=classes)

                index_sample += 1

                start_for_tokenize -= args.shift

                index_sample += 1

                start_for_tokenize -= args.shift
