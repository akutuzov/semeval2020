# /usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


shares = [0.065, 0.108, 0.122, 0.150, 0.312]
mean_ipms = [29.371, 118.305, 66.991, 380.763, 17.721]
std_ipms = [31.631, 168.215, 101.974, 278.381, 50.12]
entropy = [2.951, 3.311, 4.258, 3.448, 3.429]
above_mean = [14, 14, 40, 20, 22]
above_mean_norm = [0.452, 0.378, 0.408, 0.5, 0.458]
external_ipms = [50.077,144.657, 56.311, 179.845, 14.492]
external_std_ipms = [175.856, 274.171, 94.059, 331.353, 46.789]

# English, Swedish, German, GEMS, Latin
median = [0.2, 0.203, 0.266, 0.267, 0.364]

elmo_cos = [0.254, 0.252, 0.740, 0.323, 0.360]
elmo_apd = [0.605, 0.569, 0.560, 0.323, -0.113]
bert_cos = [0.225, 0.185, 0.590, 0.394, 0.561]
bert_apd = [0.546, 0.254, 0.427, 0.243, 0.372]

measure = median

for method in [elmo_cos, elmo_apd, bert_cos, bert_apd]:
    print('=====')
    print('Spearman correlation: %0.2f %0.2f' % spearmanr(measure, method))
    print('Pearson correlation: %0.2f %0.2f' % pearsonr(measure, method))

figure, ax = plt.subplots()
ax.plot(measure, elmo_cos, marker='o', color='black', label='ELMo PRT')
ax.plot(measure, elmo_apd, marker='o', color='red', label='ELMo APD')
ax.plot(measure, bert_cos, marker='s', color='black', label='BERT PRT')
ax.plot(measure, bert_apd, marker='s', color='red', label='BERT APD')

ax.set(xlabel='Median gold_internal score', ylabel='Performance (Spearman)',
           title='Distribution of gold_internal scores and the performance of algorithms')
ax.grid()
ax.legend(loc='best')
figure.savefig('performance.png', dpi=300)
plt.close()

