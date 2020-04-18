# /usr/bin/env python3
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
from smart_open import open
from sklearn.preprocessing import minmax_scale
import numpy as np


def rand_jitter(arr):
    stdev = .005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def degree_plot(scores_data, fname, add_title=''):
    figure, ax = plt.subplots()
    for element in sorted(scores_data, reverse=True):
        x_data = minmax_scale([float(el[1]) for el in scores_data[element]])
        much_shifted = [val for val in x_data if val > 0.6]
        much_shifted_ratio = len(much_shifted) / len(x_data)
        print('%s: %0.3f of words with change degree > 0.6' % (element, much_shifted_ratio))

        x_data = rand_jitter(x_data)
        y_data = [element] * len(x_data)
        ax.scatter(x_data, y_data, s=12, label=element)

    ax.set(xlabel='Degree of change (normalized)', ylabel='Datasets',
           title='Distribution of shift degrees across target words' + add_title)
    ax.grid()
    # ax.legend(loc='best')
    figure.savefig(fname, dpi=300)
    plt.close()
    return figure


if __name__ == '__main__':
    testset_dir = sys.argv[1]
    testset_files = [f for f in os.listdir(testset_dir) if f.endswith('.txt')]

    data = {}
    for f in testset_files:
        lang = f.split('.')[0]
        lines = [l.strip() for l in open(os.path.join(testset_dir, f), 'r').readlines()]
        data[lang] = []
        for l in lines:
            data[lang].append(l.split('\t'))

    fig = degree_plot(data, 'degree_distribution.png')

    if len(sys.argv) > 2:
        corpora_dir = sys.argv[2]
        frequencies = []
        for language in sorted(data, reverse=True):
            print('Counting frequencies for %s...' % language)
            ipms = []
            words = {el[0]: 0 for el in data[language]}
            size = 0
            corpus = os.path.join(corpora_dir, language, 'corpus.txt.gz')
            for line in open(corpus, 'r'):
                sentence = line.strip().split()
                size += len(sentence)
                for word in sentence:
                    if word in words:
                        words[word] += 1
            for el in data[language]:
                ipm = words[el[0]] / (size / 1000000)
                ipms.append(ipm)
            frequencies.append(ipms)
            print(language, np.mean(ipms))
            print([round(i, 2) for i in ipms])

        fig, ax = plt.subplots()

        ax.boxplot(frequencies, labels=sorted(data, reverse=True), whis='range')
        title = 'Word frequencies'
        xlabel = ''
        ylabel = 'Istances per million (IPM)'
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid()
        fig.savefig('ipm.png', dpi=300)
