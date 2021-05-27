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
| Morphology   |   0.595 |  0.521 | 0.525 |   0.581 | 0.555 |
| Syntax       |   0.541 |  0.646 | 0.575 |   0.645 | 0.602 |
| Average      |   0.568 |  0.583 | 0.475 |   0.710 | 0.584 |
|--------------|---------|--------|-------|---------|-------|
| JSD (tiny improvement only for task1)                     |
| Morphology   |   0.486 |  0.479 | 0.575 |   0.645 | 0.546 |
| Syntax       |   0.486 |  0.604 | 0.475 |   0.645 | 0.553 |
| Average      |   0.514 |  0.604 | 0.500 |   0.742 | 0.590 |
|--------------|---------|--------|-------|---------|-------|
| FILTERING: 5% threshold  (almost no change)	            |
| Morphology   |   0.595 |  0.521 | 0.525 |   0.581 | 0.555 |
| Syntax       |   0.541 |  0.646 | 0.525 |   0.710 | 0.605 |
| Average      |   0.568 |  0.583 | 0.450 |   0.710 | 0.578 |
|--------------|---------|--------|-------|---------|-------|
| SYNTAX feature filtering (nothing for this task)          |
| none         |   0.541 |  0.646 | 0.575 |   0.645 | 0.602 |
| group        |   0.541 |  0.646 | 0.525 |   0.645 | 0.589 |
| partial      |   0.541 |  0.646 | 0.575 |   0.645 | 0.602 |
| delete       |   0.595 |  0.521 | 0.106 |   0.581 | 0.451 |
|--------------|---------|--------|-------|---------|-------|
| SYNTAX feature filtering + 5% thr                         |
| none         |   0.541 |  0.646 | 0.525 |   0.710 | 0.605 |
| group        |   0.541 |  0.646 | 0.475 |   0.774 | 0.609 |
| partial      |   0.541 |  0.646 | 0.575 |   0.710 | 0.618 |
| delete       |   0.595 |  0.562 | 0.106 |   0.581 | 0.461 |
|--------------|---------|--------|-------|---------|-------|
| SE Count Bas |   0.595 |  0.688 | 0.525 |   0.645 | 0.613 |
| SE 1st       |   0.622 |  0.750 | 0.700 |   0.677 | 0.687 |



## Subtask 2
|              | English | German | Latin | Swedish |  Mean |
|--------------|---------|--------|-------|---------|-------|
| Morphology   |   0.234 |  0.043 | 0.241 |   0.207 | 0.181 |
| Syntax       |   0.319 |  0.163 | 0.328 |  -0.017 | 0.198 |
| Average      |   0.293 |  0.147 | 0.304 |   0.088 | 0.208 |
|--------------|---------|--------|-------|---------|-------|
| JSD (slightly worse)                                      |
| Morphology   |   0.149 |  0.054 | 0.240 |   0.251 | 0.173 |
| Syntax       |   0.232 |  0.127 | 0.343 |  -0.051 | 0.163 |
| Average      |   0.134 |  0.129 | 0.269 |   0.067 | 0.150 |
|--------------|---------|--------|-------|---------|-------|
| FILTERING: 5% threshold                                   |
| Morphology   |   0.211 |  0.080 | 0.285 |   0.191 | 0.192 |
| Syntax       |   0.331 |  0.146 | 0.265 |   0.184 | 0.231 |
| Average      |   0.315 |  0.171 | 0.345 |   0.263 | 0.273 |
|--------------|---------|--------|-------|---------|-------|
| SYNTAX feature filtering (en la improve ge sv disaster) 
| none         |   0.319 |  0.163 | 0.328 |  -0.017 | 0.168 |
| group        |   0.319 |  0.107 | 0.170 |  -0.070 | 0.131 |
| partial      |   0.328 |  0.093 | 0.382 |  -0.037 | 0.191 |
| delete       |   0.150 | -0.068 | 0.106 |  -0.034 | 0.038 |
|--------------|---------|--------|-------|---------|-------|
| SYNTAX feature filtering  + 5% thr                        |
| none         |   0.331 |  0.146 | 0.265 |   0.184 | 0.231 |
| group        |   0.331 |  0.100 | 0.096 |   0.179 | 0.176 |
| partial      |   0.343 |  0.118 | 0.321 |   0.149 | 0.233 |
| delete       |   0.157 |  0.026 | 0.104 |  -0.026 | 0.065 |
|--------------|---------|--------|-------|---------|-------|
| SE Count Bas |   0.022 |  0.216 | 0.359 |  -0.022 | 0.144 |
| SE 1st       |   0.422 |  0.725 | 0.412 |   0.547 | 0.527 |

