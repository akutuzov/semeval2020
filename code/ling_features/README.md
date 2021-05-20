# Linguistic features

- `stanza_process.py` produces parsed CONLL files from raw texts;
- `collect_ling_stats.py` reads a CONLL file and dumps frequencies for morphological and syntax properties of the target words into JSON files;
- `compare_ling.py` produces TSV files with binary and graded change predictions from two JSON files;
- `merge.py` reads two JSON files (e.g. with morphological and syntax properties) and produces averaged predictions.

Use `../eval.py` to evaluate the resulting TSVs with regards to gold scores (in the `test_data_truth` subdirectory).

# Results

## Subtask 1
These are with positive class coefficient 0.43 (top 43% of most changed words are assigned the 1 label):

|              | English | German | Latin | Swedish |  Mean |
|--------------|---------|--------|-------|---------|-------|
| Morphology   |   0.541 |  0.521 | 0.525 |   0.581 | 0.542 |
| Syntax       |   0.541 |  0.646 | 0.575 |   0.645 | 0.602 |
| Average      |   0.568 |  0.583 | 0.475 |   0.710 | 0.584 |
|--------------|---------|--------|-------|---------|-------|
| SE Count Bas |   0.595 |  0.688 | 0.525 |   0.645 | 0.613 |
| SE 1st       |   0.622 |  0.750 | 0.700 |   0.677 | 0.687 |



## Subtask 2
|              | English | German | Latin | Swedish |  Mean |
|--------------|---------|--------|-------|---------|-------|
| Morphology   |   0.239 |  0.043 | 0.241 |   0.207 | 0.183 |
| Syntax       |   0.196 |  0.163 | 0.328 |  -0.017 | 0.168 |
| Average      |   0.245 |  0.147 | 0.304 |   0.088 | 0.196 |
|--------------|---------|--------|-------|---------|-------|
| SE Count Bas |   0.022 |  0.216 | 0.359 |  -0.022 | 0.144 |
| SE 1st       |   0.422 |  0.725 | 0.412 |   0.547 | 0.527 |

