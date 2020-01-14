# Contextualised Word Representations with BERT

### Code

Under `code/bert` we implement lexical semantic change detection using BERT representations. It involves multiple steps:

1. `collect.py`: collection of contextualised representations
2. `distance.py`: quantification of semantic change (*not yet implemented*)

First, we collect all contextualised representations for the target lemmas. Then we measure the degree of change 
undergone by a lemma in terms of the geometric properties of its representations.

The script `run-bert.sh` will run, for now, only the collection phase on the [trial data](#markdown-header-trial-data). 
For this, assuming you are working on a UNIX-based system, first make the script executable with

	chmod 755 run-bert.sh

Then execute

	bash -e run-bert.sh

The script will unzip the data, iterate over corpora of each language, collect contextualised representations, store them under `matrices/`.
<!--
and write the results for the trial targets under `results/`. It will also produce answer files for task 1 and 2 in the required submission format from the results and store them under `results/`. It does this in the following way: FD and CNT+CI+CD predict change values for the target words. These values provide the ranking for task 2. Then, target words are assigned into two classes depending on whether their predicted change values exceed a specified threshold or not. If the script throws errors, you might need to install Python dependencies: `pip3 install -r requirements.txt`.
-->

### Trial Data <a name="markdown-header-trial-data"></a>

We provide trial data in `trial_data_public.zip`. For each language, it contains:

- trial target words for which predictions can be submitted in the practice phase (`targets/`)
- the true classification of the trial target words for task 1 in the practice phase, i.e., the file against which submissions will be scored in the practice phase (`truth/task1/`)
- the true ranking of the trial target words for task 2 in the practice phase (`truth/task2/`)
- a sample submission for the trial target words in the above-specified format (`answer.zip/`)
- two trial corpora from which you may predict change scores for the trial target words (`corpora/`)

__Important__: The scores in `truth/task1/` and `truth/task2/` are not meaningful as they were randomly assigned.

You can start by uploading the zipped answer folder to the system to check the submission and evaluation format. Find more information on the submission format on [the shared task website](https://languagechange.org/semeval/).

#### Trial Corpora ####

The trial corpora under `corpora/` are gzipped samples from the corpora that will be used in the evaluation phase. For each language two time-specific corpora are provided. Participants are required to predict the lexical semantic change of the target words between these two corpora. Each line contains one sentence and has the form

	lemma1 lemma2 lemma3...

Sentences have been randomly shuffled. The corpora have the same format as the ones which will be used in the evaluation phase. Find more information about the corpora on [the shared task website](https://languagechange.org/semeval/).


References <a name="references"></a>
--------

Dominik Schlechtweg, Anna Hätty, Marco del Tredici, and Sabine Schulte im Walde. 2019. [A Wind of Change: Detecting and Evaluating Lexical Semantic Change across Times and Domains](https://www.aclweb.org/anthology/papers/P/P19/P19-1072/). In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Florence, Italy. ACL.