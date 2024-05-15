#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import gc
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

import h5py
import numpy as np
import pandas as pd
import pysam

from transformers import AutoTokenizer
import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

def split_info_fields(info_record):
    records = info_record.split(";")
    result = {}
    for record in records:
        record_data = record.split("=")
        result[record_data[0]] = record_data[1]
    return result


# Get part of DNA sequence
def tokenize_sequence(
    start_for_tokenize_d,
    end_for_tokenize_d,
    chr_name,
    reference,
    tokenizer,
):
    start_seq_d = start_for_tokenize_d
    end_seq_d = end_for_tokenize_d   
 
    DNA_seq = reference.fetch(reference=chr_name, start=start_seq_d, end=end_seq_d)
    DNA_seq = DNA_seq.upper()

    # Tokenize part of DNA sequence
    tokens = tokenizer.encode_plus(
        DNA_seq, add_special_tokens=False, return_offsets_mapping=True
    )
    return tokens

def search_exons_end_cds(trans_content, strand):

    # first and last exon - for search intron
    exon_cotent = trans_content[trans_content["type"] == "exon"]
    if exon_cotent.empty:
        exon_start_d = -1000
        exon_end_d = -1000
    else:
        exon_start_d = exon_cotent["start"].values.min()
        exon_end_d = exon_cotent["end"].values.max()

    # first and last CDS - for search UTR
    cds_cotent = trans_content[trans_content["type"] == "CDS"]
    if cds_cotent.empty:
        cds_start_d = -1000
        cds_end_d = -1000
    else:
        cds_start_d = cds_cotent["start"].values.min()
        cds_end_d = cds_cotent["end"].values.max()

    return exon_start_d, exon_end_d, cds_start_d, cds_end_d


#classification
def classification(
    tokens_for_classing,
    trans_content,
    exon_start_d,
    exon_end_d,
    cds_start_d,
    cds_end_d,
    start_for_tokenize_d,
    strand
):
    def process_transcipt_element(tr, arr):
        for t in ["exon", "CDS"]:
            if tr["type"] == t:
                arr[
                    class_lables.index(t),
                    tr["start"] + DNA2tokens_shift : tr["end"] + DNA2tokens_shift,
                ] = 1

    for i in [exon_start_d, exon_end_d, start_for_tokenize_d]:
        assert i >= 0

    DNA2tokens_shift = - start_for_tokenize_d
    assert DNA2tokens_shift <= 0

    tokens2DNA_shift = - DNA2tokens_shift
    assert tokens2DNA_shift >= 0

    tokens_start_d = tokens_for_classing["offset_mapping"][0][0] + tokens2DNA_shift
    tokens_end_d = tokens_for_classing["offset_mapping"][-1][-1] + tokens2DNA_shift
    assert tokens_end_d > tokens_start_d

    tokens_for_classing = np.array(tokens_for_classing["offset_mapping"])

    class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS", "intergenic"]

    classes_t = np.zeros(
        shape=(len(class_lables), tokens_end_d - tokens_start_d), dtype=np.int8
    )

    exon_start_t = exon_start_d + DNA2tokens_shift

    if exon_start_t > 0:
        classes_t[class_lables.index("intergenic"), :exon_start_t] = 1

    exon_end_t = exon_end_d + DNA2tokens_shift
    if exon_end_t < tokens_end_d - tokens_start_d - 1:
        classes_t[
            class_lables.index("intergenic"),
            exon_end_t:] = 1

    trans_content.apply(process_transcipt_element, arr=classes_t, axis="columns")

    classes_t[class_lables.index("intron"), exon_start_t:exon_end_t] = (
        classes_t[class_lables.index("intergenic"), exon_start_t:exon_end_t]
        + classes_t[class_lables.index("exon"), exon_start_t:exon_end_t]
    ) == 0

    if cds_start_d > 0:
        cds_start_t = cds_start_d + DNA2tokens_shift
        assert exon_start_t <= cds_start_t, "CDS start is not within exon"
        if exon_start_t < cds_start_t:
            if strand == "+":
                classes_t[class_lables.index("5UTR"), exon_start_t:cds_start_t] = classes_t[class_lables.index("exon"), exon_start_t:cds_start_t]
            if strand == "-":
                classes_t[class_lables.index("3UTR"), exon_start_t:cds_start_t] = classes_t[class_lables.index("exon"), exon_start_t:cds_start_t]


    if cds_end_d > 0:
        cds_end_t = cds_end_d + DNA2tokens_shift
        assert cds_end_t <= exon_end_t, "CDS end is not within exon"
        if cds_end_t < exon_end_t:
            if strand == "+":
                classes_t[class_lables.index("3UTR"), cds_end_t + 1 : exon_end_t + 1] = classes_t[class_lables.index("exon"), cds_end_t + 1 : exon_end_t + 1]
            if strand == "-":
                classes_t[class_lables.index("5UTR"), cds_end_t + 1 : exon_end_t + 1] = classes_t[class_lables.index("exon"), cds_end_t + 1 : exon_end_t + 1]


    classes = np.zeros(
        shape=(len(class_lables), len(tokens_for_classing)), dtype=np.int8
    )

    for ind, (st, end) in enumerate(tokens_for_classing):
        classes[:, ind] = classes_t[
            :, st:end
        ].max(axis=1)

    return classes


# Main process
def process_transcript(transcript, index_transcript):
        logging.debug(f"------------new transcript start------------")
        transcript_name = transcript
        transcript_content = data_grouped.get_group(transcript)
        transcript_info = transcripts_index.loc[transcript]
        gene = transcript_info["Parent"]
        transcript_strand = transcript_info["strand"]
        transcript_chr = transcript_info["seqid"]
        transcript_class = transcript_info["type"]


        start_for_tokenize_d = transcript_info["start"] - 2000
        end_for_tokenize_d = transcript_info["end"] + 2000

        assert (
            transcript_strand == "+" or transcript_strand == "-"
        ), "Not identifited strand"

        (
            exon_start_d,
            exon_end_d,
            cds_start_d,
            cds_end_d,
        ) = search_exons_end_cds(transcript_content, transcript_strand)

        transcript_stats = [
            transcript_chr,
            gene,
            transcript_name,
            trans_class_dict[transcript_class],
            transcript_strand,
        ]

        logging.debug(f"tokenize start")
        transcript_tokens = tokenize_sequence(
            start_for_tokenize_d,
            end_for_tokenize_d,
            transcript_chr,
            ref,
            tokenizer,
        )

        logging.debug(f"classification start")
        classes = classification(
            transcript_tokens,
            transcript_content,
            exon_start_d,
            exon_end_d,
            cds_start_d,
            cds_end_d,
            start_for_tokenize_d,
            transcript_strand
        )

        coordinates = [int(start_for_tokenize_d), int(end_for_tokenize_d)]

        with h5py.File(args.output, "a") as file:
            group = file.create_group(f"transcript_{index_transcript}")
            group.create_dataset("info", data=transcript_stats)
            group.create_dataset(
                "coordinates", data=coordinates
            )
            group.create_dataset("input_ids", data=np.array(transcript_tokens["input_ids"]))
            group.create_dataset("labels", data=classes)
            group.create_dataset("token_type_ids", data=np.array(transcript_tokens["token_type_ids"]))
            group.create_dataset("attention_mask", data=np.array(transcript_tokens["attention_mask"]))


            del classes
            del transcript_tokens
            gc.collect()

            index_transcript += 1

        return index_transcript


# Let's do it
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chrs", type=str, default="all", help="Path to TXT file with chromosomes' names"
    )
    parser.add_argument("--gff", type=str, help="Path to GFF file")
    parser.add_argument("--fasta", type=str, help="Path to FASTA file")
    parser.add_argument("--fai", type=str, help="Path to FAI file")
    parser.add_argument("--output", type=str, help="Path to output HDF5 file")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="AIRI-Institute/gena-lm-bigbird-base-t2t",
        help="tokenizer",
    )
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
    new_cols = pd.DataFrame.from_records(data["attributes"].apply(split_info_fields))
    data = pd.concat((data, new_cols), axis=1)

    data_grouped = data.groupby("Parent")

    data_lnc_RNA = data["type"] == "lnc_RNA"
    data_mRNA = data["type"] == "mRNA"
    transcript_name_mask = data_mRNA | data_lnc_RNA

    valid_transcripts = data[transcript_name_mask]["ID"].unique()

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

    index_transcript = 0
    index_limit = 100
    for transcript in tqdm(valid_transcripts):
      index_transcript = process_transcript(transcript, index_transcript)
      if index_transcript >= index_limit:
        del tokenizer
        gc.collect()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        index_limit += 100
