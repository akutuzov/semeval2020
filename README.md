# Starting kit (Evaluation Phase)
Starting kit for [SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948).

The code draws from the [LSCDetection repository](https://github.com/Garrafao/LSCDetection).

### Code

Under `code/` we provide an implementation of the two baselines for the shared task:

1. normalized frequency difference ([FD](https://github.com/Garrafao/LSCDetection))
2. count vectors with column intersection and cosine distance ([CNT+CI+CD](https://github.com/Garrafao/LSCDetection))

FD first calculates the frequency for each target word in each of the two corpora, normalizes it by the total corpus frequency and then calculates the absolute difference in these values as a measure of change. CNT+CI+CD first learns vector representations for each of the two corpora, then aligns them by intersecting their columns and measures change by cosine distance between the two vectors for a target word.

The script `run.sh` will run FD and CNT+CI+CD on the test data. For this, assuming you are working on a UNIX-based system, first make the script executable with

	chmod 755 run.sh

Then execute

	bash -e run.sh

The script will download and unzip the test data, iterate over the corpora of each language, learn matrices, store them under `matrices/` and write the results for the test targets under `results/`. It will also produce answer files for task 1 and 2 in the required submission format from the results and store them under `results/`. It does this in the following way: FD and CNT+CI+CD predict change values for the target words. These values provide the ranking for task 2. Then, target words are assigned into two classes depending on whether their predicted change values exceed a specified threshold or not. If the script throws errors, you might need to install Python dependencies: `pip3 install -r requirements.txt`.

### Test Data

Running `run.sh` as described above will automatically download the test data. The test data can also be manually downloaded here:

- [English](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/sem-eval-ulscd-eng/)
- [German](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/sem-eval-ulscd-ger/)
- [Latin](https://zenodo.org/record/3674099)
- [Swedish](https://zenodo.org/record/3672950)

It has the same format as the trial data with the difference that no true classification and ranking is provided. For each language we provide the target words and a corpus pair.

In the starting kit we also provide a sample submission for the test target words with randomly assigned scores (`sample_answer.zip`).


Reference
--------

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky and Nina Tahmasebi. [SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948). To appear in SemEval@COLING2020.
