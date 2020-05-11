# Contextualised Embeddings for Lexical Semantic Change Detection
This code accompanies the paper `*[UiO-UvA at SemEval-2020 Task 1: Contextualised Embeddings for Lexical Semantic Change Detection](https://arxiv.org/abs/2005.00050)*',
which describes our participation in SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection.

##  Extraction of contextualized token embeddings

For ELMo: `python3 code/elmo/extract_elmo.py --input <CORPUS> --elmo <ELMO_MODEL> --outfile <OUTFILE> --vocab <TARGET_WORDS>`

For BERT: `python3 code/bert/collect.py <PATH_TO_MODEL> <CORPUS> <TARGET_WORDS> <OUTFILE>`

These scripts produce `npz` archives containing numpy arrays with token embeddings for each target word in a given corpus.

## Estimating semantic change
- COS algorithm: `python3 code/cosine.py -t <TARGET_WORDS> -i0 corpus0.npz -i1 corpus1.npz > cosine_change.txt`

- APD algorithm: `python3 code/distance.py <TARGET_WORDS> corpus0.npz corpus1.npz apd_change.txt`

- JSD algorithm: `python3 code/jsd.py <TARGET_WORDS> corpus0.npz corpus1.npz jsd_change.txt`

These scripts produce plain text files containing lists of words with their corresponding degree of semantic change between
*corpus0* and *corpus1*.

## Download pre-trained embeddings

### ELMo
- [English](http://vectors.nlpl.eu/repository/20/209.zip)
- [German](http://vectors.nlpl.eu/repository/20/201.zip)
- [Latin](http://vectors.nlpl.eu/repository/20/203.zip)
- [Swedish](http://vectors.nlpl.eu/repository/20/202.zip)

### BERT
- [English](https://huggingface.co/bert-base-uncased)
- [German](https://huggingface.co/bert-base-german-cased)
- [Latin](https://huggingface.co/bert-base-multilingual-cased)
- [Swedish](https://huggingface.co/af-ai-center/bert-large-swedish-uncased)

## Authors
- Andrey Kutuzov (University of Oslo, Norway)
- Mario Giulianelli (University of Amsterdam, Netherlands)


### SemEval-2020 Task 1 Reference
--------

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky and Nina Tahmasebi.
[SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948).
To appear in SemEval@COLING2020.
