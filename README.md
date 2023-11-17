# GeneContent
Перед запуском: 
1. В строке #10 data = pd.read_csv("/storage3/calengil/GENA_dataset/data/GCF_009914755.1/genomic.gff", sep='\t', header=None, skiprows=9) нужно прописать путь к файлу GFF.
2. В строке #55 ref = pysam.Fastafile("/storage3/calengil/GENA_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna") нужно прописать путь к fasta.

Результат просто принтуется, поэтому нужно самостоятельно направлять output в файл.
Число входных токенов inp и сдвиг рамки shift заданы в самом начале (пока не придумал, куда их засунуть как гиперпараметры)

Приложен также файл head_109_genomic.gff. Это первые 109 строк genomic.gff, которые можно использовать в качестве теста. В нем 8 полных генов и 1 обрезанный.
