#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModel
import pysam
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

data = pd.read_csv("/storage3/calengil/GENA_dataset/data/GCF_009914755.1/genomic.gff", sep='\t', header=None, skiprows=9)

inp = 512 #count of model input tokens
shift = inp / 2



#Masks for classes
data_gene = data[2] == 'gene'
data_pre_miRNA = data[2] == 'primary_transcript'
data_lnc_RNA = data[2] == 'lnc_RNA'
data_miRNA = data[2] == 'miRNA'
data_miscRNA = data[2] == 'transcript'
data_snoRNA = data[2] == 'snoRNA'
data_exon = data[2] == 'exon'
data_CDS = data[2] == 'CDS'
data_mRNA = data[2] == 'mRNA'

trans_class_dict = {'gene': 'gene', 'primary_transcript': 'pre_miRNA', 'lnc_RNA': 'lnc_RNA', 'miRNA': 'miRNA', 'transcript': 'miscRNA', 'snoRNA': 'snoRNA', 'exon': 'exon', 'CDS': 'CDS', 'mRNA': 'mRNA'}

#Search genes names
def search_genes_names(data_for_search):
  names = []
  genes_info = data[data_gene][8].tolist()
  for i in range(len(genes_info)):
      attribute = genes_info[i].split(';')
      names.append(attribute[0].split('=')[1] + ';')
  return names

genes_names = search_genes_names(data)
############################################

#Search transcripts names for gene
list_transcript_names = []
transcript_name = ''
transcript_name_mask = data_miscRNA | data_mRNA | data_pre_miRNA | data_lnc_RNA | data_snoRNA | data_miRNA
for gene in genes_names:
  transcripts_of_gene = []
  for transcript in data[transcript_name_mask][8]:
    if gene in transcript:
      transcript_name = transcript.split(';')[0]
      transcripts_of_gene.append(transcript_name.split('=')[1])
  list_transcript_names.append(transcripts_of_gene)

#Reference and length
ref = pysam.Fastafile("/storage3/calengil/GENA_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna")
L = inp * 8

############################################

#Transcript's content to list
def get_trans_content(trans_name):
  transcript_content_mask = data[8].str.contains(trans_name)
  transcript_content_list = data[transcript_content_mask].values.tolist()
  return transcript_content_list

#Get part of DNA sequence
def seq_transcript(trans_name, radius):
  DNA_seq = ref.fetch(reference=get_trans_content(trans_name)[0][0], start=(start_for_tokenize - radius*2), end=(start_for_tokenize + radius*2))
  DNA_seq = DNA_seq.upper()
  return DNA_seq


def find_center_in_mapping_tokens(map, point):
  for interval in map:
    if interval[0] <= point <= interval[1]:
      return interval
  return None

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-large-t2t')

##################################################################################################
#Tokenize part of DNA sequence (forward)
def tokenize_forward_seq(seq):
  #Tokenize sequence
  tokens = tokenizer.tokenize(seq)
  token_to_chars = tokenizer.encode_plus(seq, add_special_tokens=False, return_offsets_mapping=True)
  mapping_tokens = token_to_chars['offset_mapping']
  
  while len(mapping_tokens) < inp:
    new_radius = L + 250
    seq = seq_transcript(transcript_name, new_radius) 
    tokens = tokenizer.tokenize(seq)   
    token_to_chars = tokenizer.encode_plus(seq, add_special_tokens=False, return_offsets_mapping=True)
    mapping_tokens = token_to_chars['offset_mapping']


  #Select part of sequence
  center = int(mapping_tokens[-1][-1] / 2)
  central_token = find_center_in_mapping_tokens(mapping_tokens, center)

  tokens_for_classing = [central_token]
  count_of_tokens = 1
  right_len = central_token[1] - center
  left_len = center - central_token[0]
  total_len = right_len + left_len

  while count_of_tokens < inp:
    if right_len > left_len:
      tokens_for_classing.insert(0, mapping_tokens[mapping_tokens.index(tokens_for_classing[0]) - 1])
      left_len = center - tokens_for_classing[0][0]
      count_of_tokens += 1
    else:
      tokens_for_classing.append(mapping_tokens[mapping_tokens.index(tokens_for_classing[-1]) + 1])
      right_len = tokens_for_classing[-1][1] - center
      count_of_tokens += 1
  total_len = right_len + left_len

  def tok_coor_to_DNA_coor(token_coor):
      return start_for_tokenize - center + token_coor ####################

  #first and last exon - for search intron
  first_exon_end = -1000
  for part in range(len(get_trans_content(transcript_name))):
    if get_trans_content(transcript_name)[part][2] == 'exon':
      last_exon_start = get_trans_content(transcript_name)[part][3]
      last_exon_end = get_trans_content(transcript_name)[part][4]
      if first_exon_end != -1000:
        continue
      first_exon_end = get_trans_content(transcript_name)[part][4]
      first_exon_start = get_trans_content(transcript_name)[part][3]

  #first and last CDS - for search UTR
  first_cds_start = -1000
  for part in range(len(get_trans_content(transcript_name))):
    if get_trans_content(transcript_name)[part][2] == 'CDS':
      last_cds_end = get_trans_content(transcript_name)[part][4]
      if first_cds_start != -1000:
        continue
      first_cds_start = get_trans_content(transcript_name)[part][3]

  #Classes
  class_0 = []
  class_1 = []
  class_2 = []
  class_3 = []
  class_4 = []

  #prior classify tokens
  for i in range(len(tokens_for_classing)):
    token_start = tok_coor_to_DNA_coor(tokens_for_classing[i][0])
    token_end = tok_coor_to_DNA_coor(tokens_for_classing[i][1])
    prior_token_class = []
    for part in range(len(get_trans_content(transcript_name))):
      if get_trans_content(transcript_name)[part][3] <= token_end and get_trans_content(transcript_name)[part][4] >= token_start:
        if get_trans_content(transcript_name)[part][2] == 'exon':
          prior_token_class.append('exon')
          if token_start < get_trans_content(transcript_name)[part][3] and first_exon_end != get_trans_content(transcript_name)[part][4]:             #Search introns
            prior_token_class.append('intron')
          elif token_end > get_trans_content(transcript_name)[part][4] and last_exon_start != get_trans_content(transcript_name)[part][3]:
            prior_token_class.append('intron')
          if first_cds_start != -1000:
            if first_cds_start > first_exon_start:
              if token_start < first_cds_start:                                                          #Search UTR
                prior_token_class.append('5\'UTR')
            if last_cds_end < last_exon_end:
              if token_end > last_cds_end:
                prior_token_class.append('3\'UTR')
        if get_trans_content(transcript_name)[part][2] == 'CDS':
           prior_token_class.append('CDS')
    if 'exon' not in prior_token_class and token_start > first_exon_end and token_end < last_exon_start:
      prior_token_class.append('intron')
    if '5\'UTR' in prior_token_class:
      class_0.append(1)
    else:
      class_0.append(0)
    if 'exon' in prior_token_class:
      class_1.append(1)
    else:
      class_1.append(0)
    if 'intron' in prior_token_class:
      class_2.append(1)
    else:
      class_2.append(0)
    if '3\'UTR' in prior_token_class:
      class_3.append(1)
    else:
      class_3.append(0)
    if 'CDS' in prior_token_class:
      class_4.append(1)
    else:
      class_4.append(0)

  token_ids = token_to_chars["input_ids"]
  token_types = token_to_chars["token_type_ids"]
  attention_mask = token_to_chars["attention_mask"]

  return tok_coor_to_DNA_coor(tokens_for_classing[0][0]), tok_coor_to_DNA_coor(tokens_for_classing[-1][1]), token_ids, token_types, attention_mask, class_0, class_1, class_2, class_3, class_4

####################################################################################################################################################################
#Tokenize part of DNA sequence (reverse)
def tokenize_reverse_seq(seq):
  #Tokenize sequence
  tokens = tokenizer.tokenize(seq)
  token_to_chars = tokenizer.encode_plus(seq, add_special_tokens=False, return_offsets_mapping=True)
  mapping_tokens = token_to_chars['offset_mapping']
  
  while len(mapping_tokens) < inp:
    new_radius = L + 250
    seq = seq_transcript(transcript_name, new_radius) 
    tokens = tokenizer.tokenize(seq)   
    token_to_chars = tokenizer.encode_plus(seq, add_special_tokens=False, return_offsets_mapping=True)
    mapping_tokens = token_to_chars['offset_mapping']

  #Select part of sequence
  center = int(mapping_tokens[-1][-1] / 2)
  central_token = find_center_in_mapping_tokens(mapping_tokens, center)

  tokens_for_classing = [central_token]
  count_of_tokens = 1
  right_len = central_token[1] - center
  left_len = center - central_token[0]
  total_len = right_len + left_len

  while count_of_tokens < inp:
    if right_len > left_len:
      tokens_for_classing.insert(0, mapping_tokens[mapping_tokens.index(tokens_for_classing[0]) - 1])
      left_len = center - tokens_for_classing[0][0]
      count_of_tokens += 1
    else:
      tokens_for_classing.append(mapping_tokens[mapping_tokens.index(tokens_for_classing[-1]) + 1])
      right_len = tokens_for_classing[-1][1] - center
      count_of_tokens += 1
  total_len = right_len + left_len

  def tok_coor_to_DNA_coor(token_coor):
      return start_for_tokenize - center + token_coor ####################

  #first and last exon - for search intron
  first_exon_end = -1000
  for part in range(len(get_trans_content(transcript_name))):
    if get_trans_content(transcript_name)[part][2] == 'exon':
      last_exon_start = get_trans_content(transcript_name)[part][4]
      last_exon_end = get_trans_content(transcript_name)[part][3]
      if first_exon_end != -1000:
        continue
      first_exon_end = get_trans_content(transcript_name)[part][3]
      first_exon_start = get_trans_content(transcript_name)[part][4]


  #first and last CDS - for search UTR
  first_cds_start = -1000
  for part in range(len(get_trans_content(transcript_name))):
    if get_trans_content(transcript_name)[part][2] == 'CDS':
      last_cds_end = get_trans_content(transcript_name)[part][3]
      if first_cds_start != -1000:
        continue
      first_cds_start = get_trans_content(transcript_name)[part][4]

  #Classes
  class_0 = []
  class_1 = []
  class_2 = []
  class_3 = []
  class_4 = []

  #prior classify tokens
  for i in range(len(tokens_for_classing)):
    token_start = tok_coor_to_DNA_coor(tokens_for_classing[i][0])
    token_end = tok_coor_to_DNA_coor(tokens_for_classing[i][1])
    prior_token_class = []
    for part in range(len(get_trans_content(transcript_name))):
      if get_trans_content(transcript_name)[part][3] <= token_end and get_trans_content(transcript_name)[part][4] >= token_start:
        if get_trans_content(transcript_name)[part][2] == 'exon':
          prior_token_class.append('exon')
          if token_start < get_trans_content(transcript_name)[part][3] and last_exon_start != get_trans_content(transcript_name)[part][4]:     #Search introns
            prior_token_class.append('intron')
          if token_end > get_trans_content(transcript_name)[part][4] and first_exon_end != get_trans_content(transcript_name)[part][3]:
            prior_token_class.append('intron')
          if first_cds_start != -1000:
            if first_cds_start < first_exon_start:
              if token_end > first_cds_start:                                                          #Search UTR
                prior_token_class.append('5\'UTR')
            if last_cds_end > last_exon_end:
              if token_start < last_cds_end:
                prior_token_class.append('3\'UTR')
        if get_trans_content(transcript_name)[part][2] == 'CDS':
           prior_token_class.append('CDS')
    if 'exon' not in prior_token_class and token_start > last_exon_start and token_end < first_exon_end:
      prior_token_class.append('intron')
    if '5\'UTR' in prior_token_class:
      class_0.append(1)
    else:
      class_0.append(0)
    if 'exon' in prior_token_class:
      class_1.append(1)
    else:
      class_1.append(0)
    if 'intron' in prior_token_class:
      class_2.append(1)
    else:
      class_2.append(0)
    if '3\'UTR' in prior_token_class:
      class_3.append(1)
    else:
      class_3.append(0)
    if 'CDS' in prior_token_class:
      class_4.append(1)
    else:
      class_4.append(0)

  token_ids = token_to_chars["input_ids"]
  token_types = token_to_chars["token_type_ids"]
  attention_mask = token_to_chars["attention_mask"]

  return tok_coor_to_DNA_coor(tokens_for_classing[0][0]), tok_coor_to_DNA_coor(tokens_for_classing[-1][1]), token_ids, token_types, attention_mask, class_0, class_1, class_2, class_3, class_4

####################################################################################################################################################################
for gene in genes_names:
  for transcript_name in list_transcript_names[genes_names.index(gene)]:
    for transcript_number in range(len(get_trans_content(transcript_name))):
      if 'ID=' + transcript_name in get_trans_content(transcript_name)[transcript_number][8]:
        if get_trans_content(transcript_name)[transcript_number][6] == '+':
          transcript_start = get_trans_content(transcript_name)[transcript_number][3]
          transcript_end = get_trans_content(transcript_name)[transcript_number][4]
          transcript_class = get_trans_content(transcript_name)[transcript_number][2]
          transcript_chr = get_trans_content(transcript_name)[transcript_number][0]
          transcript_strand = get_trans_content(transcript_name)[transcript_number][6]
        if get_trans_content(transcript_name)[transcript_number][6] == '-':
          transcript_start = get_trans_content(transcript_name)[transcript_number][4]
          transcript_end = get_trans_content(transcript_name)[transcript_number][3]
          transcript_class = get_trans_content(transcript_name)[transcript_number][2]
          transcript_chr = get_trans_content(transcript_name)[transcript_number][0]
          transcript_strand = get_trans_content(transcript_name)[transcript_number][6]

    start_for_tokenize = transcript_start
    end_out = transcript_end
    start_out = transcript_end

    if transcript_strand == '+':
      while end_out <= transcript_end:
        start_out, end_out, token_ids, token_types, attention_mask, class_0, class_1, class_2, class_3, class_4 = tokenize_forward_seq(seq_transcript(transcript_name, L))
        print(f'{gene[:-1]}\t{transcript_name}\t{trans_class_dict[transcript_class]}\t{transcript_chr}\t{transcript_strand}\t{int(start_out)}\t{int(end_out)}\t{token_ids}\t{token_types}\t{attention_mask}\t{class_0}\t{class_1}\t{class_2}\t{class_3}\t{class_4}')
        start_for_tokenize += shift
    if transcript_strand == '-':
      while start_out >= transcript_end:
        start_out, end_out, token_ids, token_types, attention_mask, class_0, class_1, class_2, class_3, class_4 = tokenize_reverse_seq(seq_transcript(transcript_name, L))
        print(f'{gene[:-1]}\t{transcript_name}\t{trans_class_dict[transcript_class]}\t{transcript_chr}\t{transcript_strand}\t{int(start_out)}\t{int(end_out)}\t{token_ids}\t{token_types}\t{attention_mask}\t{class_0}\t{class_1}\t{class_2}\t{class_3}\t{class_4}')
        start_for_tokenize -= shift



