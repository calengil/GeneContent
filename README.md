# GeneContent
## search_gene_content.py

Dependecies: argparse, h5py, numpy, pandas, pysam, transformers.

Usage: $ python3 search_gene_content.py --chrs=[chromosomes] --gff=[gff] --fasta=[fasta] --fai=[fai] --inp=[tokens] --output=[output] --shift=[shift] --tokenizer_name=[tokenizer_name] --radius=[radius] 

**[chromosomes]**: enter path to txt file with chosen chromosomes names

**[gff]**: enter path to gff file

**[fasta]**: enter path to fasta file

**[fai]**: enter path to fai or chrom.sizes file

**[tokens]**: enter number of input tokens (default 512)

**[output]**: enter path to hdf5 file

**[shift]**: enter step between tokenization points (default 256)

**[tokenizer_name]**: enter your tokenizer (default "AIRI-Institute/gena-lm-bert-large-t2t")

**[radius]**: enter coefficient for choose part of DNA (default 64)

There are 2 groups in the output hdf5 file:

*/run_info/info* contains an array with 8 elements (run inforamation: gff file, fasta file, fai file, number of input tokens, hdf5 file, shift, tokenizer, radius)

*/records* contains groups of samples named after transcript names. Example: */records/rna-NR_024321.1/sample_0*

There are 6 subgroups in each *sample_X*:

  */records/transcript_name/sample_X/info* contains an array with 5 elements (information about the sample: [chromosome, gene name, transcript name, transcript type, strand])
  
  */records/transcript_name/sample_X/coordinates* contains an array with 2 elements (part coordinates: [start, end])

  */records/transcript_name/sample_X/token_ids* contains token_ids array

  */records/transcript_name/sample_X/token_types* contains token_types_ids array

  */records/transcript_name/sample_X/attention_mask* contains attention_mask array

  */records/transcript_name/sample_X/classes* contains an array with  the classification of tokens (array size = count of classes x number of input tokens)

## decode_to_bed.py (now not supported)

Dependecies: argparse, pandas, transformers.

Usage: $ python3 decode_to_bed.py --tokenized_seqs=[txt] > output.bed

**[txt]**: enter path to search_gene_content.py output file

## GenePredictionDataset.py

Dependecies: numpy, torch.utils.data, h5py.


