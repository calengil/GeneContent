# GeneContent
## search_gene_content.py

Dependecies: argparse, h5py, numpy, pandas, pysam, transformers.

Usage: $ python3 search_gene_content.py --chrs=[chromosomes] --gff=[gff] --fasta=[fasta] --fai=[fai] --output=[output] --tokenizer_name=[tokenizer_name]

**[chromosomes]**: enter path to txt file with chosen chromosomes names

**[gff]**: enter path to gff file

**[fasta]**: enter path to fasta file

**[fai]**: enter path to fai or chrom.sizes file

**[output]**: enter path to hdf5 file

**[tokenizer_name]**: enter your tokenizer (default "AIRI-Institute/gena-lm-bert-large-t2t")

There are some groups in the output hdf5 file:

  */sample_X/info* contains an array with 5 elements (information about the sample: [chromosome, gene name, transcript name, transcript type, strand])
  
  */sample_X/coordinates* contains an array with 2 elements (part coordinates: [start, end])

  */sample_X/input_ids* contains token_ids array

  */sample_X/token_type_ids* contains token_types_ids array

  */sample_X/attention_mask* contains attention_mask array

  */sample_X/labels* contains an array with  the classification of tokens (array size = count of classes x number of input tokens)

## decode_to_bed.py

Dependecies: argparse, h5py, pandas, transformers.

Usage: $ python3 decode_to_bed.py --tokenized_seqs=[hdf5] > output.bed

**[hdf5]**: enter path to search_gene_content.py output file

## short_GenePredictionDataset.py

Dependecies: numpy, torch.utils.data, h5py.


