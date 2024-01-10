#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import h5py
import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--chrs", type=str, default="all", help="Path to TXT file with chromosomes' names"
)
parser.add_argument("--gff", type=str, help="Path to GFF file")
parser.add_argument("--fasta", type=str, help="Path to FASTA file")
parser.add_argument("--fai", type=str, help="Path to FAI file")
parser.add_argument("--inp", type=int, default=512, help="number of input tokens")
parser.add_argument("--output", type=str, help="Path to output HDF5 file")
parser.add_argument("--shift", type=int, default=-1, help="shift")
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default="AIRI-Institute/gena-lm-bert-large-t2t",
    help="tokenizer",
)
parser.add_argument("--radius", type=int, default=64, help="radius")
args = parser.parse_args()

if args.shift == -1:
    shift = "half the length of the DNA segment"
else:
    shift = args.shift

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
data = pd.read_csv(args.gff, sep="\t", names=col_names, header=None, comment="#")
data["start"] = (
    data["start"] - 1
)  # coodinates in GFF file are 1-based, convert to 0-based
# data["end"] = data["end"] - 1 # do not substract from the end; intervals in GFF are closed, but now we can consider
#                               them as half-opened intervals

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


special_tokens = get_service_token_encodings(tokenizer)


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
    trans_name,
    start_for_tokenize_d,
    tokens_count,
    chr_name,
    chr_info,
    reference,
    radius,
):
    start_seq_d = start_for_tokenize_d - radius * tokens_count
    end_seq_d = start_for_tokenize_d + radius * tokens_count

    chr_length = chr_info[chr_info[0] == chr_name][1].values[0]

    if start_seq_d < 0:
        start_seq_d = 0
    elif end_seq_d > chr_length:
        end_seq_d = chr_length

    assert (
        end_seq_d < chr_length and start_seq_d >= 0
    ), "DNA beyond the boundaries of the chromosome"

    DNA_seq = reference.fetch(reference=chr_name, start=start_seq_d, end=end_seq_d)
    DNA_seq = DNA_seq.upper()

    # Tokenize part of DNA sequence
    tokens = tokenizer.tokenize(DNA_seq)
    token_to_chars = tokenizer.encode_plus(
        DNA_seq, add_special_tokens=False, return_offsets_mapping=True
    )
    mapping_tokens = token_to_chars["offset_mapping"]

    while len(mapping_tokens) < args.inp - 2:
        if start_seq_d > 0:
            if start_seq_d - 500 >= 0:
                start_seq_d = start_seq_d - 500
            else:
                start_seq_d = 0
        if end_seq_d < chr_length:
            if end_seq_d + 500 <= chr_length:
                end_seq_d = end_seq_d + 500
            else:
                end_seq_d = chr_length

        DNA_seq = reference.fetch(reference=chr_name, start=start_seq_d, end=end_seq_d)
        DNA_seq = DNA_seq.upper()

        token_to_chars = tokenizer.encode_plus(
            DNA_seq, add_special_tokens=False, return_offsets_mapping=True
        )
        mapping_tokens = token_to_chars["offset_mapping"]

        if start_seq_d == 0 and end_seq_d == chr_length:
            break

    # Find start for tokenize for coordinates of tokens
    center_s = start_for_tokenize_d - start_seq_d
    return token_to_chars, center_s


def select_part(token_to_chars, center_s, inp):
    # Select part of sequence
    mapping_tokens = token_to_chars["offset_mapping"]

    assert len(mapping_tokens) == len(set(mapping_tokens))
    assert (
        mapping_tokens[0][0] <= center_s <= mapping_tokens[-1][-1]
    ), "no center_s in tokens"

    for interval in mapping_tokens:
        if interval[0] <= center_s <= interval[1]:
            central_token = interval

    tokens_for_classing = [central_token]
    count_of_tokens = 1
    right_len = central_token[1] - center_s
    left_len = center_s - central_token[0]

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
                    left_len = center_s - tokens_for_classing[0][0]
                    count_of_tokens += 1
                else:
                    tokens_for_classing.append(
                        mapping_tokens[
                            mapping_tokens.index(tokens_for_classing[-1]) + 1
                        ]
                    )
                    right_len = tokens_for_classing[-1][1] - center_s
                    count_of_tokens += 1
            elif right_len < left_len:
                if tokens_for_classing[-1] != mapping_tokens[-1]:
                    tokens_for_classing.append(
                        mapping_tokens[
                            mapping_tokens.index(tokens_for_classing[-1]) + 1
                        ]
                    )
                    right_len = tokens_for_classing[-1][1] - center_s
                    count_of_tokens += 1
                else:
                    tokens_for_classing.insert(
                        0,
                        mapping_tokens[
                            mapping_tokens.index(tokens_for_classing[0]) - 1
                        ],
                    )
                    left_len = center_s - tokens_for_classing[0][0]
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
        first_exon_start_d = -1000
        last_exon_end_d = -1000
    else:
        if strand == "+":
            first_exon_start_d = exon_cotent["start"].values.min()
            last_exon_end_d = exon_cotent["end"].values.max()
        elif strand == "-":
            first_exon_start_d = exon_cotent["end"].values.max()
            last_exon_end_d = exon_cotent["start"].values.min()

    # first and last CDS - for search UTR
    cds_cotent = trans_content[trans_content["type"] == "CDS"]
    if cds_cotent.empty:
        first_cds_start_d = -1000
        last_cds_end_d = -1000
    else:
        if strand == "+":
            first_cds_start_d = cds_cotent["start"].values.min()
            last_cds_end_d = cds_cotent["end"].values.max()
        elif strand == "-":
            first_cds_start_d = cds_cotent["end"].values.max()
            last_cds_end_d = cds_cotent["start"].values.min()

    return first_exon_start_d, last_exon_end_d, first_cds_start_d, last_cds_end_d


# classification + strand
def classification_forward(
    tokens_for_classing,
    trans_content,
    first_exon_start_d,
    last_exon_end_d,
    first_cds_start_d,
    last_cds_end_d,
    start_for_tokenize_d,
    center_s,
):
    def process_transcipt_element(tr, arr):
        for t in ["exon", "CDS"]:
            if tr["type"] == t:
                arr[
                    class_lables.index(t),
                    tr["start"] + DNA2targetshift : tr["end"] + DNA2targetshift,
                ] = 1

    for i in [first_exon_start_d, last_exon_end_d, start_for_tokenize_d, center_s]:
        assert i >= 0

    segment2DNAshift = start_for_tokenize_d - center_s
    assert segment2DNAshift >= 0

    segment2targetshift = -tokens_for_classing[0][0]
    assert segment2targetshift <= 0
    assert (
        tokens_for_classing[0][0] + segment2targetshift == 0
    ), f"tokens_for_classing[0][0]: {tokens_for_classing[0][0]}, segment2targetshift {segment2targetshift}"

    DNA2targetshift = -segment2DNAshift + segment2targetshift
    assert DNA2targetshift <= 0

    tokens_start_d = tokens_for_classing[0][0] + segment2DNAshift
    tokens_end_d = tokens_for_classing[-1][-1] + segment2DNAshift
    assert tokens_end_d > tokens_start_d

    tokens_for_classing = np.array(tokens_for_classing)

    class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic"]

    classes_t = np.zeros(
        shape=(len(class_lables), tokens_end_d - tokens_start_d), dtype=np.int8
    )

    first_exon_start_t = first_exon_start_d + DNA2targetshift

    if first_exon_start_t > 0:
        classes_t[class_lables.index("intergenic"), :first_exon_start_t] = 1

    last_exon_end_t = last_exon_end_d + DNA2targetshift
    if last_exon_end_t < tokens_end_d - tokens_start_d - 1:
        classes_t[
            class_lables.index("intergenic"),
            last_exon_end_t + 1 : tokens_end_d - tokens_start_d,
        ] = 1

    trans_content.apply(process_transcipt_element, arr=classes_t, axis="columns")

    classes_t[class_lables.index("intron"), :] = (
        classes_t[class_lables.index("intergenic")]
        + classes_t[class_lables.index("exon")]
    ) == 0

    if first_cds_start_d > 0:
        first_cds_start_t = first_cds_start_d + DNA2targetshift
        assert first_exon_start_t <= first_cds_start_t, "CDS start is not within exon"
        if first_exon_start_t < first_cds_start_t:
            classes_t[
                class_lables.index("5UTR"), first_exon_start_t:first_cds_start_t
            ] = 1

    if last_cds_end_d > 0:
        last_cds_end_t = last_cds_end_d + DNA2targetshift
        assert last_cds_end_t <= last_exon_end_t, "CDS end is not within exon"
        if last_cds_end_t < last_exon_end_t:
            classes_t[
                class_lables.index("3UTR"), last_cds_end_t + 1 : last_exon_end_t + 1
            ] = 1
    classes = np.zeros(
        shape=(len(class_lables), len(tokens_for_classing) + 2), dtype=np.int8
    )

    classes[:, 0] = -100
    classes[:, -1] = -100

    for ind, (st, end) in enumerate(tokens_for_classing):
        classes[:, ind + 1] = classes_t[
            :, st + segment2targetshift : end + segment2targetshift
        ].max(axis=1)

    return classes


# classification - strand
def classification_reverse(
    tokens_for_classing,
    trans_content,
    first_exon_start_d,
    last_exon_end_d,
    first_cds_start_d,
    last_cds_end_d,
    start_for_tokenize_d,
    center_s,
):
    def process_transcipt_element(tr, arr):
        for t in ["exon", "CDS"]:
            if tr["type"] == t:
                arr[
                    class_lables.index(t),
                    tr["start"] + DNA2targetshift : tr["end"] + DNA2targetshift,
                ] = 1

    for i in [first_exon_start_d, last_exon_end_d, start_for_tokenize_d, center_s]:
        assert i >= 0

    segment2DNAshift = start_for_tokenize_d - center_s
    assert segment2DNAshift >= 0

    segment2targetshift = -tokens_for_classing[0][0]
    assert segment2targetshift <= 0
    assert (
        tokens_for_classing[0][0] + segment2targetshift == 0
    ), f"tokens_for_classing[0][0]: {tokens_for_classing[0][0]}, segment2targetshift {segment2targetshift}"

    DNA2targetshift = -segment2DNAshift + segment2targetshift
    assert DNA2targetshift <= 0

    tokens_start_d = tokens_for_classing[0][0] + segment2DNAshift
    tokens_end_d = tokens_for_classing[-1][-1] + segment2DNAshift
    assert tokens_end_d > tokens_start_d

    tokens_for_classing = np.array(tokens_for_classing)

    class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic"]

    classes_t = np.zeros(
        shape=(len(class_lables), tokens_end_d - tokens_start_d), dtype=np.int8
    )

    first_exon_start_t = first_exon_start_d + DNA2targetshift
    last_exon_end_t = last_exon_end_d + DNA2targetshift

    if last_exon_end_t > 0:
        classes_t[class_lables.index("intergenic"), :last_exon_end_t] = 1  # done

    if first_exon_start_t < tokens_end_d - tokens_start_d - 1:
        classes_t[
            class_lables.index("intergenic"),
            first_exon_start_t + 1 : tokens_end_d - tokens_start_d,
        ] = 1  # done

    trans_content.apply(
        process_transcipt_element, arr=classes_t, axis="columns"
    )  # done?

    classes_t[class_lables.index("intron"), :] = (
        classes_t[class_lables.index("intergenic")]
        + classes_t[class_lables.index("exon")]
    ) == 0  # done?

    if first_cds_start_d > 0:
        first_cds_start_t = first_cds_start_d + DNA2targetshift
        assert first_exon_start_t >= first_cds_start_t, "CDS start is not within exon"
        if first_exon_start_t > first_cds_start_t:
            classes_t[
                class_lables.index("5UTR"),
                first_cds_start_t + 1 : first_exon_start_t + 1,
            ] = 1  # done

    if last_cds_end_d > 0:
        last_cds_end_t = last_cds_end_d + DNA2targetshift
        assert last_cds_end_t >= last_exon_end_t, "CDS end is not within exon"
        if last_cds_end_t > last_exon_end_t:
            classes_t[class_lables.index("3UTR"), last_exon_end_t:last_cds_end_t] = 1

    classes = np.zeros(
        shape=(len(class_lables), len(tokens_for_classing) + 2), dtype=np.int8
    )

    classes[:, 0] = -100
    classes[:, -1] = -100

    for ind, (st, end) in enumerate(tokens_for_classing):
        classes[:, ind + 1] = classes_t[
            :, st + segment2targetshift : end + segment2targetshift
        ].max(axis=1)

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
        "shift=" + str(shift),
        "tokenizer=" + args.tokenizer_name,
        "radius=" + str(args.radius),
    ]
    array_info = run_info.create_dataset("info", data=array_run_info)

    transcripts_records = file.create_group("records")
    for transcript in tqdm(valid_transcripts):
        index_sample = 0
        group = transcripts_records.create_group(transcript)
        transcript_name = transcript
        transcript_content = data_grouped.get_group(transcript)  # TODO change to apply
        transcript_info = transcripts_index.loc[transcript]
        gene = transcript_info["Parent"]
        transcript_strand = transcript_info["strand"]
        transcript_chr = transcript_info["seqid"]
        transcript_class = transcript_info["type"]

        if transcript_strand == "+":
            transcript_start_d = transcript_info["start"]
            transcript_end_d = transcript_info["end"]
        elif transcript_strand == "-":
            transcript_start_d = transcript_info["end"]
            transcript_end_d = transcript_info["start"]

        assert (
            transcript_strand == "+" or transcript_strand == "-"
        ), "Not identifited strand"

        (
            first_exon_start_d,
            last_exon_end_d,
            first_cds_start_d,
            last_cds_end_d,
        ) = search_exons_end_cds(transcript_content, transcript_strand)

        start_for_tokenize_d = transcript_start_d
        end_out_d = transcript_end_d
        start_out_d = transcript_end_d

        transcript_stats = [
            transcript_chr,
            gene,
            transcript_name,
            trans_class_dict[transcript_class],
            transcript_strand,
        ]

        if transcript_strand == "+":
            while end_out_d <= transcript_end_d:
                transcript_part_tokens, center_s = tokenize_sequence(
                    transcript_name,
                    start_for_tokenize_d,
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
                    transcript_part_tokens, center_s, args.inp
                )

                classes = classification_forward(
                    selected_tokens_coor,
                    transcript_content,
                    first_exon_start_d,
                    last_exon_end_d,
                    first_cds_start_d,
                    last_cds_end_d,
                    start_for_tokenize_d,
                    center_s,
                )

                start_out_d = (
                    start_for_tokenize_d - center_s + selected_tokens_coor[0][0]
                )
                end_out_d = (
                    start_for_tokenize_d - center_s + selected_tokens_coor[-1][1]
                )
                first_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[0])
                last_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[-1])

                token_ids, token_types, attention_mask = select_tokens(
                    transcript_part_tokens,
                    first_selected_token_index,
                    last_selected_token_index,
                    special_tokens,
                )
                coordinates = [int(start_out_d), int(end_out_d)]

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
                if args.shift == -1:
                    start_for_tokenize_d += (
                        selected_tokens_coor[-1][-1] - selected_tokens_coor[0][0]
                    ) // 2
                else:
                    start_for_tokenize_d += args.shift

        elif transcript_strand == "-":
            while start_out_d >= transcript_end_d:
                transcript_part_tokens, center_s = tokenize_sequence(
                    transcript_name,
                    start_for_tokenize_d,
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
                    transcript_part_tokens, center_s, args.inp
                )

                classes = classification_reverse(
                    selected_tokens_coor,
                    transcript_content,
                    first_exon_start_d,
                    last_exon_end_d,
                    first_cds_start_d,
                    last_cds_end_d,
                    start_for_tokenize_d,
                    center_s,
                )

                start_out_d = (
                    start_for_tokenize_d - center_s + selected_tokens_coor[0][0]
                )
                end_out_d = (
                    start_for_tokenize_d - center_s + selected_tokens_coor[-1][1]
                )
                first_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[0])
                last_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[-1])

                token_ids, token_types, attention_mask = select_tokens(
                    transcript_part_tokens,
                    first_selected_token_index,
                    last_selected_token_index,
                    special_tokens,
                )
                coordinates = [int(start_out_d), int(end_out_d)]

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
                if args.shift == -1:
                    start_for_tokenize_d -= (
                        selected_tokens_coor[-1][-1] - selected_tokens_coor[0][0]
                    ) // 2
                else:
                    start_for_tokenize_d -= args.shift
