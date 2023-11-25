#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd
import pysam
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--gff", type=str, help="Input dir for GFF file")
parser.add_argument("--fasta", type=str, help="Input dir for FASTA file")
parser.add_argument("--fai", type=str, help="Input dir for FAI file")
parser.add_argument("--inp", type=int, default=512, help="Input number of input tokens")
parser.add_argument("--shift", type=int, default=256, help="Input shift")
args = parser.parse_args()

data = pd.read_csv(args.gff, sep="\t", header=None, skiprows=9)
data = data.dropna()  #!!!!!!!!!
ref = pysam.Fastafile(args.fasta)
tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-large-t2t")
chromsizes = pd.read_csv(args.fai, sep="\t", header=None)

trans_class_dict = {
    "gene": "gene",
    "primary_transcript": "pre_miRNA",
    "lnc_RNA": "lnc_RNA",
    "miRNA": "miRNA",
    "transcript": "miscRNA",
    "snoRNA": "snoRNA",
    "exon": "exon",
    "CDS": "CDS",
    "mRNA": "mRNA",
}


# Search genes names
def search_genes_names(data_for_search):
    data_gene = data_for_search[2] == "gene"
    names = []
    genes_info = data_for_search[data_gene][8].tolist()
    for i in range(len(genes_info)):
        attribute = genes_info[i].split(";")
        names.append(attribute[0].split("=")[1] + ";")
    assert len(names) != 0, "No names of genes"
    return names


# Search transcripts names for gene
def search_transcripts_names(data_for_search, genes_names):
    transcripts_names = []
    transcript_name = ""
    data_pre_miRNA = data_for_search[2] == "primary_transcript"
    data_lnc_RNA = data_for_search[2] == "lnc_RNA"
    data_miRNA = data_for_search[2] == "miRNA"
    data_miscRNA = data_for_search[2] == "transcript"
    data_snoRNA = data_for_search[2] == "snoRNA"
    data_mRNA = data_for_search[2] == "mRNA"
    transcript_name_mask = (
        data_miscRNA
        | data_mRNA
        | data_pre_miRNA
        | data_lnc_RNA
        | data_snoRNA
        | data_miRNA
    )
    for gene in genes_names:
        transcripts_of_gene = []
        for transcript in data_for_search[transcript_name_mask][8]:
            if gene in transcript:
                transcript_name = transcript.split(";")[0]
                transcripts_of_gene.append(transcript_name.split("=")[1])
        assert len(transcripts_of_gene) > 0, "Gene without any transcript"
        transcripts_names.append(transcripts_of_gene)
    return transcripts_names


# Transcript's content to dataframe
def get_trans_content(trans_name, data_for_search):
    transcript_content_mask = data_for_search[8].str.contains(trans_name)
    content_df = data_for_search[transcript_content_mask]
    return content_df


# Get part of DNA sequence
def tokenize_sequence(
    trans_name, start_for_tokenize, radius, chr_name, chr_info, reference
):
    start_seq = start_for_tokenize - radius * 8 * 2
    end_seq = start_for_tokenize + radius * 8 * 2

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

    while len(mapping_tokens) < args.inp:
        if start_seq > 0:
            if start_seq + 500 >= 0:
                start_seq = start_seq + 500
            else:
                start_seq = 0
        if end_seq < chr_length:
            if end_seq + 500 <= chr_length:
                end_seq = end_seq + 500
            else:
                end_seq = chr_length

        DNA_seq = reference.fetch(reference=chr_name, start=start_seq, end=end_seq)
        DNA_seq = DNA_seq.upper()

        tokens = tokenizer.tokenize(DNA_seq)
        token_to_chars = tokenizer.encode_plus(
            DNA_seq, add_special_tokens=False, return_offsets_mapping=True
        )
        mapping_tokens = token_to_chars["offset_mapping"]
    # Find start for tokenize for coordinates of tokens
    center = start_for_tokenize - start_seq
    return token_to_chars, center


def select_part(token_to_chars, center, inp):
    # Select part of sequence
    mapping_tokens = token_to_chars["offset_mapping"]

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

    while count_of_tokens < inp:
        if right_len >= left_len and tokens_for_classing[0] != mapping_tokens[0]:
            tokens_for_classing.insert(
                0, mapping_tokens[mapping_tokens.index(tokens_for_classing[0]) - 1]
            )
            left_len = center - tokens_for_classing[0][0]
            count_of_tokens += 1
        elif right_len < left_len and tokens_for_classing[0] != mapping_tokens[-1]:
            tokens_for_classing.append(
                mapping_tokens[mapping_tokens.index(tokens_for_classing[-1]) + 1]
            )
            right_len = tokens_for_classing[-1][1] - center
            count_of_tokens += 1

    assert len(tokens_for_classing) == inp, "Number of tokens is too few"

    return tokens_for_classing


def search_exons_end_cds(trans_content, strand):
    # first and last exon - for search intron
    exon_cotent = trans_content[trans_content[2] == "exon"]
    if exon_cotent.empty:
        first_exon_start = -1000
        last_exon_end = -1000
    else:
        if strand == "+":
            first_exon_start = exon_cotent[3].values.min()
            last_exon_end = exon_cotent[4].values.max()
        elif strand == "-":
            first_exon_start = exon_cotent[4].values.max()
            last_exon_end = exon_cotent[3].values.min()

    # first and last CDS - for search UTR
    cds_cotent = trans_content[trans_content[2] == "CDS"]
    if cds_cotent.empty:
        first_cds_start = -1000
        last_cds_end = -1000
    else:
        if strand == "+":
            first_cds_start = cds_cotent[3].values.min()
            last_cds_end = cds_cotent[4].values.max()
        elif strand == "-":
            first_cds_start = cds_cotent[4].values.max()
            last_cds_end = cds_cotent[3].values.min()

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
    class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS"]
    classes = np.zeros(shape=(len(class_lables), len(tokens_for_classing)), dtype=int)
    final_token_class = []

    # prior classify tokens
    for i, tok in enumerate(tokens_for_classing):
        token_start = start_for_tokenize - center + tok[0]
        token_end = start_for_tokenize - center + tok[1]
        prior_token_class = []
        if token_start <= last_exon_end and token_end >= first_exon_start:
            find_exon = (
                (trans_content[2] == "exon")
                & (trans_content[3] <= token_end)
                & (trans_content[4] >= token_start)
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
                    (trans_content[2] == "exon")
                    & (trans_content[3] > token_start)
                    & (trans_content[3] <= token_end)
                    & (trans_content[3] > first_exon_start)
                )
                if not trans_content[on_left_of_exon].empty:
                    prior_token_class.append("intron")
                on_right_of_exon = (
                    (trans_content[2] == "exon")
                    & (trans_content[4] >= token_start)
                    & (trans_content[4] < token_end)
                    & (trans_content[4] < last_exon_end)
                )
                if not trans_content[on_right_of_exon].empty:
                    prior_token_class.append("intron")
            # Search CDS
            find_cds = (
                (trans_content[2] == "CDS")
                & (trans_content[3] <= token_end)
                & (trans_content[4] >= token_start)
            )
            if not trans_content[find_cds].empty:
                prior_token_class.append("CDS")
        final_token_class.append(prior_token_class)

    for tok_index, tok in enumerate(tokens_for_classing):
        for l_index, label in enumerate(class_lables):
            if label in final_token_class[tok_index]:
                classes[l_index, tok_index] = 1

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
    class_lables = ["5UTR", "exon", "intron", "3UTR", "CDS"]
    classes = np.zeros(shape=(len(class_lables), len(tokens_for_classing)), dtype=int)
    final_token_class = []

    # prior classify tokens
    for i, tok in enumerate(tokens_for_classing):
        token_start = start_for_tokenize - center + tok[0]
        token_end = start_for_tokenize - center + tok[1]
        prior_token_class = []
        if token_start <= first_exon_start and token_end >= last_exon_end:
            find_exon = (
                (trans_content[2] == "exon")
                & (trans_content[3] <= token_end)
                & (trans_content[4] >= token_start)
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
                    (trans_content[2] == "exon")
                    & (trans_content[3] > token_start)
                    & (trans_content[3] <= token_end)
                    & (trans_content[3] > last_exon_end)
                )
                if not trans_content[on_left_of_exon].empty:
                    prior_token_class.append("intron")
                on_right_of_exon = (
                    (trans_content[2] == "exon")
                    & (trans_content[4] >= token_start)
                    & (trans_content[4] < token_end)
                    & (trans_content[4] < first_exon_start)
                )
                if not trans_content[on_right_of_exon].empty:
                    prior_token_class.append("intron")
            # Search CDS
            find_cds = (
                (trans_content[2] == "CDS")
                & (trans_content[3] <= token_end)
                & (trans_content[4] >= token_start)
            )
            if not trans_content[find_cds].empty:
                prior_token_class.append("CDS")
        final_token_class.append(prior_token_class)

    for tok_index, tok in enumerate(tokens_for_classing):
        for l_index, label in enumerate(class_lables):
            if label in final_token_class[tok_index]:
                classes[l_index, tok_index] = 1

    return classes


# Let's do it

genes_names_list = search_genes_names(data)
transcripts_names_list = search_transcripts_names(data, genes_names_list)

for gene in genes_names_list:
    for transcript_name in transcripts_names_list[genes_names_list.index(gene)]:
        trans_content_df = get_trans_content(transcript_name, data)
        transcript_info = trans_content_df[
            trans_content_df[8].str.contains("ID=" + transcript_name)
        ]
        transcript_strand = transcript_info[6].values[0]
        transcript_chr = transcript_info[0].values[0]
        transcript_class = transcript_info[2].values[0]
        if transcript_strand == "+":
            transcript_start = transcript_info[3].values[0]
            transcript_end = transcript_info[4].values[0]
        elif transcript_strand == "-":
            transcript_start = transcript_info[4].values[0]
            transcript_end = transcript_info[3].values[0]
        assert (
            transcript_strand == "+" or transcript_strand == "-"
        ), "Not identifited strand"

        (
            first_exon_start,
            last_exon_end,
            first_cds_start,
            last_cds_end,
        ) = search_exons_end_cds(trans_content_df, transcript_strand)
        start_for_tokenize = transcript_start
        end_out = transcript_end
        start_out = transcript_end

        if transcript_strand == "+":
            while end_out <= transcript_end:
                transcript_part_tokens, center = tokenize_sequence(
                    transcript_name,
                    start_for_tokenize,
                    args.inp,
                    transcript_chr,
                    chromsizes,
                    ref,
                )
                selected_tokens_coor = select_part(
                    transcript_part_tokens, center, args.inp
                )
                classes = classification_forward(
                    selected_tokens_coor,
                    trans_content_df,
                    first_exon_start,
                    last_exon_end,
                    first_cds_start,
                    last_cds_end,
                    start_for_tokenize,
                    center,
                )
                start_out = start_for_tokenize - center + selected_tokens_coor[0][0]
                end_out = start_for_tokenize - center + selected_tokens_coor[-1][1]
                token_ids = transcript_part_tokens["input_ids"]
                token_types = transcript_part_tokens["token_type_ids"]
                attention_mask = transcript_part_tokens["attention_mask"]
                first_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[0])
                last_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[-1])
                print(
                    f"{gene[:-1]}\t{transcript_name}\t{trans_class_dict[transcript_class]}\t{transcript_chr}\t{transcript_strand}\t{int(start_out)}\t{int(end_out)}\t{token_ids[first_selected_token_index:last_selected_token_index + 1]}\t{token_types[first_selected_token_index:last_selected_token_index + 1]}\t{attention_mask[first_selected_token_index:last_selected_token_index + 1]}"
                )
                print(classes[0].tolist())
                print(classes[1].tolist())
                print(classes[2].tolist())
                print(classes[3].tolist())
                print(classes[4].tolist())
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
                )
                selected_tokens_coor = select_part(
                    transcript_part_tokens, center, args.inp
                )
                classes = classification_reverse(
                    selected_tokens_coor,
                    trans_content_df,
                    first_exon_start,
                    last_exon_end,
                    first_cds_start,
                    last_cds_end,
                    start_for_tokenize,
                    center,
                )
                start_out = start_for_tokenize - center + selected_tokens_coor[0][0]
                end_out = start_for_tokenize - center + selected_tokens_coor[-1][1]
                token_ids = transcript_part_tokens["input_ids"]
                token_types = transcript_part_tokens["token_type_ids"]
                attention_mask = transcript_part_tokens["attention_mask"]
                first_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[0])
                last_selected_token_index = transcript_part_tokens[
                    "offset_mapping"
                ].index(selected_tokens_coor[-1])
                print(
                    f"{gene[:-1]}\t{transcript_name}\t{trans_class_dict[transcript_class]}\t{transcript_chr}\t{transcript_strand}\t{int(start_out)}\t{int(end_out)}\t{token_ids[first_selected_token_index:last_selected_token_index + 1]}\t{token_types[first_selected_token_index:last_selected_token_index + 1]}\t{attention_mask[first_selected_token_index:last_selected_token_index + 1]}"
                )
                print(classes[0].tolist())
                print(classes[1].tolist())
                print(classes[2].tolist())
                print(classes[3].tolist())
                print(classes[4].tolist())
                start_for_tokenize -= args.shift
