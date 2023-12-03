# GeneContent
## search_gene_content.py

Dependecies: argparse, numpy, pandas, pysam, transformers.

Usage: $ python3 search_gene_content_v4.0.py --gff=[gff] --fasta=[fasta] --fai=[fai] --inp=[tokens] --shift=[shift] --radius=[radius] > output.txt

**[gff]**: enter path to gff file

**[fasta]**: enter path to fasta file

**[fai]**: enter path to fai or chrom.sizes file

**[tokens]**: enter number of input tokens (default 512)

**[shift]**: enter step between tokenization points (default 256)

**[radius]**: enter coefficient for choose part of DNA (default 64)

## decode_to_bed.py

Dependecies: argparse, pandas, transformers.

Usage: $ python3 decode_to_bed.py --tokenized_seqs=[txt] > output.bed

**[txt]**: enter path to search_gene_content.py output file

