# GeneContent
Dependecies: argparse, numpy, pandas, pysam, transformers.

Usage: $ python3 search_gene_content_v4.0.py --gff=[gff] --fasta=[fasta] --fai=[fai] --inp=[tokens] --shift=[shift] > output.txt
*[gff]: enter path for gff file
*[fasta]: enter path for fasta file
*[fai]: enter path for fai or chrom.sizes file
*[tokens]: enter number of input tokens (default 512)
*[shift]: enter step between tokenization points (default 256)

There is also an artificial sample of gff file for testing.
