# Tracing semantic shifts for Russian
This repository features the code and the dataset related to the paper
*"[Tracing cultural diachronic semantic shifts in Russian using word embeddings: test sets and baselines](http://www.dialog-21.ru/media/4598/fominvplusetal-116.pdf)"*
by Vadim Fomin, Daria Bakshandayeva, Julia Rodina and Andrey Kutuzov
(accepted to Dialog-2019).
The slides of the presentation used during Dialog-2019 are available at [dialogue_slides.pdf](https://github.com/wadimiusz/diachrony_for_russian/blob/master/dialogue_slides.pdf).

# Dataset

The `micro.csv` file in the `datasets` directory contains Russian adjectives
manually annotated for temporal semantic shifts in the time span from 2000 to 2014.

It consists of 280 entries. 
For each entry, three annotators considered a word from the column WORD (e.g., _свиной_ 'of a swine, related to a swine') 
and decided to what degree the word in question has changed its meaning from year in the `BASE_YEAR` column 
(e.g., 2009, the year when swine flu was widely discussed in media) to the next year. 
Individual annotator scores can be found in the `ASSESSOR1`, `ASSESSOR2`, and `ASSESSOR3` columns. 
Scores are on the scale from 0 to 2; 
to calculate the final score, a simple arithmetic mean of the scores was taken
(the `ASSESSOR_MEAN` column) and rounded to the nearest integer. 
The rounded value was considered to be the ground truth (the `GROUND TRUTH` column).

The `macro.csv` file in the same directory contains 215 Russian words. 43 of them (35 nouns and 5 adjectives) are manually picked words that have undergone semantic changes from pre-Soviet through Soviet times. There also are four fillers per each target word (152 nouns and 20 adjectives). The target words are marked as 1 and the fillers are marked as 0. 

See the paper for further details of the dataset creation.

# Code

The `algos` directory contains our implementation of the semantic shift detection algorithms 
used to trace semantic shifts in Russian words:

- Jaccard distance
- Kendall tau distance
- Procrustes alignment
- Global Anchors

# Using the code

Given two embedding models you can evaluate what is the degree of semantic change
for any given word `X` (must be present in both models).

Run the `score_word.py` script as follows:

```
python3 score_word.py -w X -m1 2000.model -m2 2014.model
```
This will print out the scores according to each of the 4 algorithms
(*higher score means higher similarity* between the word meaning in two models):

```
KendallTau score: -0.05795918367346939 (from -1 to 1)
Jaccard score: 0.0 (from 0 to 1)
Global Anchors score: 0.36681556701660156 (from -1 to 1)
Procrustes aligner score: 0.17986169457435608 (from -1 to 1)
```

# Historical embedding models for Russian

The diachronic word embedding models we used in the paper
are available for downloading at https://rusvectores.org/news_history/diachrony_russian/.

# BibTex
```
@article{fomin-et-al-2019,
  title={Tracing cultural diachronic semantic shifts in {R}ussian using word embeddings: test sets and baselines},
  author={Fomin, Vadim and Bakshandaeva, Daria and Rodina, Julia and Kutuzov, Andrey},
  journal={Komp'yuternaya Lingvistika i Intellektual'nye Tekhnologii: Dialog conference},
  pages={203--218},
  url={http://www.dialog-21.ru/media/4598/fominvplusetal-116.pdf},
  year={2019}
}
```
