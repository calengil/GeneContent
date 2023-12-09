# GeneContent
## search_gene_content.py

Dependecies: argparse, h5py, numpy, pandas, pysam, transformers.

Usage: $ python3 search_gene_content.py --gff=[gff] --fasta=[fasta] --fai=[fai] --inp=[tokens] --output=[output]--shift=[shift] --tokenizer_name=[tokenizer_name] --radius=[radius] 

**[gff]**: enter path to gff file

**[fasta]**: enter path to fasta file

**[fai]**: enter path to fai or chrom.sizes file

**[tokens]**: enter number of input tokens (default 512)

**[output]**: enter path to hdf5 file

**[shift]**: enter step between tokenization points (default 256)

**[tokenizer_name]**: enter your tokenizer (default "AIRI-Institute/gena-lm-bert-large-t2t")

**[radius]**: enter coefficient for choose part of DNA (default 64)

## decode_to_bed.py (now not supported)

Dependecies: argparse, pandas, transformers.

Usage: $ python3 decode_to_bed.py --tokenized_seqs=[txt] > output.bed

**[txt]**: enter path to search_gene_content.py output file

## GenePredictionDataset.py

Dependecies: numpy, torch.utils.data, h5py.
