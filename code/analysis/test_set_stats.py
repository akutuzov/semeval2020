# /usr/bin/env python3
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
from smart_open import open
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
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
        ax.scatter(x_data, y_data, s=14, label=element)

    ax.set_xlabel('Degree of change (normalized)', fontsize=20)
    ax.set_ylabel('Datasets', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_title('Distribution of shift degrees across target words' + add_title, size=20)
    ax.grid()
    figure.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return figure


def calc_entropy(words_data):
    elements = [float(el[1]) for el in words_data]
    return entropy(elements)


def calc_bias(words_data):
    elements = [float(el[1]) for el in words_data]
    mean = np.mean(elements)
    high = [el for el in elements if el > mean]
    return len(high) / len(elements)


def calc_median(words_data):
    elements = [float(el[1]) for el in words_data]
    elements = minmax_scale(elements)
    return np.median(elements)


if __name__ == '__main__':
    testset_dir = sys.argv[1]
    testset_files = [f for f in os.listdir(testset_dir) if f.endswith('.txt')]

    data = {}
    for f in testset_files:
        lang = f.split('.')[0]
        lines = [line.strip() for line in open(os.path.join(testset_dir, f), 'r').readlines()]
        data[lang] = []
        for line in lines:
            data[lang].append(line.split('\t'))

    for lang in data:
        print(f'Entropy: {lang} {calc_entropy(data[lang]):.3f}')
        print(f'Median gold score: {lang}, {calc_median(data[lang]):.3f}')
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
            print(f'{language}, {np.mean(ipms):.3f}, {np.std(ipms):.3f}')

        fig, a = plt.subplots()

        a.boxplot(frequencies, labels=sorted(data, reverse=True), whis='range')
        title = 'Word frequencies'
        xlabel = ''
        ylabel = 'Istances per million (IPM)'
        a.set(xlabel=xlabel, ylabel=ylabel, title=title)
        a.grid()
        fig.savefig('ipm.png', dpi=300)
