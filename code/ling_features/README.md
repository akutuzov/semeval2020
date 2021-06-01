# Linguistic features

- `stanza_process.py` produces parsed CONLL files from raw texts;
- `collect_ling_stats.py` reads a CONLL file and dumps frequencies for morphological and syntax properties of the target words into JSON files;
- `compare_ling.py` produces TSV files with binary and graded change predictions from two JSON files;
- `merge.py` reads two JSON files (e.g. with morphological and syntax properties) and produces averaged predictions.

Use `../eval.py` to evaluate the resulting TSVs with regards to gold scores (in the `test_data_truth` subdirectory).

# Results

## Subtask 1
These are with positive class coefficient 0.43 (top 43% of most changed words are assigned the 1 label):

|              | English | German | Latin | Swedish | Italian |  Mean |
|--------------|---------|--------|-------|---------|---------|-------|
| Morphology   |   0.595 |  0.521 | 0.525 |   0.581 |  0.722  | 0.589 |
| Syntax       |   0.541 |  0.646 | 0.575 |   0.645 |  0.611  | 0.604 |
| Average      |   0.568 |  0.583 | 0.475 |   0.710 |  0.722  | 0.612 |
|--------------|---------|--------|-------|---------|---------|-------|
| JSD (tiny improvement only for task1)                               |
| Morphology   |   0.486 |  0.479 | 0.575 |   0.645 |  0.722  | 0.581 |
| Syntax       |   0.486 |  0.604 | 0.475 |   0.645 |  0.722  | 0.586 |
| Average      |   0.514 |  0.604 | 0.500 |   0.742 |  0.778  | 0.628 |
|--------------|---------|--------|-------|---------|---------|-------|
| FILTERING: 5% threshold  (almost no change)                        |
| Morphology   |   0.595 |  0.521 | 0.525 |   0.581 |  0.722  | 0.589 |
| Syntax       |   0.541 |  0.646 | 0.525 |   0.710 |  0.611  | 0.607 |
| Average      |   0.568 |  0.583 | 0.450 |   0.710 |  0.667  | 0.596 |
|--------------|---------|--------|-------|---------|---------|-------|
| SYNTAX feature filtering (nothing for this task)                    |
| none         |   0.541 |  0.646 | 0.575 |   0.645 |  0.611  | 0.604 |
| group        |   0.541 |  0.646 | 0.525 |   0.645 |  0.722  | 0.616 |
| partial      |   0.541 |  0.646 | 0.575 |   0.645 |  0.611  | 0.604 |
| delete       |   0.595 |  0.521 | 0.106 |   0.581 |  0.500  | 0.461 |
|--------------|---------|--------|-------|---------|---------|-------|
| SYNTAX feature filtering + 5% thr                                   |
| none         |   0.541 |  0.646 | 0.525 |   0.710 |  0.611  | 0.607 |
| group        |   0.541 |  0.646 | 0.475 |   0.774 |  0.611  | 0.609 |
| partial      |   0.541 |  0.646 | 0.575 |   0.710 |  0.611  | 0.617 |
| delete       |   0.595 |  0.562 | 0.106 |   0.581 |  0.500  | 0.469 |
|--------------|---------|--------|-------|---------|---------|-------|
| Morphological feature separation                                    |
| Morphology   |   0.595 |  0.479 | 0.575 |   0.774 |  0.500  | 0.585 |
| Average      |   0.568 |  0.562 | 0.475 |   0.742 |  0.556  | 0.581 |
|--------------|---------|--------|-------|---------|---------|-------|
| + 5% filtering                                                      |
| Morphology   |   0.595 |  0.479 | 0.575 |   0.774 |  0.500  | 0.585 |
| Average      |   0.568 |  0.604 | 0.400 |   0.806 |  0.611  | 0.598 |
|--------------|---------|--------|-------|---------|---------|-------|
| SE Count Bas |   0.595 |  0.688 | 0.525 |   0.645 |  0.611* | 0.613 |
| SE 1st       |   0.622 |  0.750 | 0.700 |   0.677 |  0.944  | 0.739 |

> \* The Italian baseline relies on collocations (Basile et al., 2019): for each target word, two vector representations are built, consisting of the Bag-of-Collocations related to the two different time periods (T0 and T1). Then, the cosine similarity between the two BoCs is computed.

## Subtask 2
|              | English | German | Latin | Swedish |  Mean |Russian1 | Russian2 | Russian3 |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
| Morphology   |   0.234 |  0.043 | 0.241 |   0.207 | 0.181 | 0.137   | 0.210    | 0.327    |
| Syntax       |   0.319 |  0.163 | 0.328 |  -0.017 | 0.198 | 0.060   | 0.101    | 0.269    |
| Average      |   0.293 |  0.147 | 0.304 |   0.088 | 0.208 | 0.101   | 0.191    | 0.294    |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
| JSD (slightly worse)                                      |
| Morphology   |   0.149 |  0.054 | 0.240 |   0.251 | 0.173 |
| Syntax       |   0.232 |  0.127 | 0.343 |  -0.051 | 0.163 |
| Average      |   0.134 |  0.129 | 0.269 |   0.067 | 0.150 |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
| FILTERING: 5% threshold                                   |
| Morphology   |   0.211 |  0.080 | 0.285 |   0.191 | 0.192 | 0.127   | 0.185    | 0.264    |
| Syntax       |   0.331 |  0.146 | 0.265 |   0.184 | 0.231 | 0.056   | 0.111    | 0.279    |
| Average      |   0.315 |  0.171 | 0.345 |   0.263 | 0.273 | 0.094   | 0.183    | 0.278    |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
| SYNTAX feature filtering (en la improve ge sv disaster)   |
| none         |   0.319 |  0.163 | 0.328 |  -0.017 | 0.168 |
| group        |   0.319 |  0.107 | 0.170 |  -0.070 | 0.131 |
| partial      |   0.328 |  0.093 | 0.382 |  -0.037 | 0.191 |
| delete       |   0.150 | -0.068 | 0.106 |  -0.034 | 0.038 |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
| SYNTAX feature filtering  + 5% thr                        |
| none         |   0.331 |  0.146 | 0.265 |   0.184 | 0.231 |
| group        |   0.331 |  0.100 | 0.096 |   0.179 | 0.176 |
| partial      |   0.343 |  0.118 | 0.321 |   0.149 | 0.233 |
| delete       |   0.157 |  0.026 | 0.104 |  -0.026 | 0.065 |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
| Morphological feature separation                          |
| Morphology   |   0.236 | -0.040 | 0.242 |   0.348 | 0.196 | 0.146   | 0.022    | 0.118    |
| Average      |   0.283 |  0.027 | 0.288 |   0.279 | 0.219 | 0.156   | 0.039    | 0.140    |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
| + 5% filtering                                            |
| Morphology   |   0.212 | -0.019 | 0.074 |  0.416  | 0.171 | 0.205   | 0.009    | 0.148    |
| Average      |   0.308 |  0.067 | 0.190 |  0.382  | 0.237 | 0.201   | 0.024    | 0.171    |
|--------------|---------|--------|-------|---------|-------|---------|----------|----------|
|    Count Bas |   0.022 |  0.216 | 0.359 |  -0.022 | 0.144 | 0.314   | 0.302    | 0.381    |
|    1st       |   0.422 |  0.725 | 0.412 |   0.547 | 0.527 | 0.798   | 0.803    | 0.822    |

