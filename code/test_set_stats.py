# /usr/bin/env python3
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
from smart_open import open
from sklearn.preprocessing import minmax_scale

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

    fig, ax = plt.subplots()
    for language in sorted(data, reverse=True):
        x_data = minmax_scale([float(el[1]) for el in data[language]])
        y_data = [language for el in x_data]
        ax.scatter(x_data, y_data, s=10, label=language)

        much_shifted = [val for val in x_data if val > 0.6]
        much_shifted_ratio = len(much_shifted) / len(x_data)
        print('%s: %0.3f of words with change degree > 0.6' % (language, much_shifted_ratio))

    ax.set(xlabel='Degree of change (normalized)', ylabel='Languages',
           title='Distribution of shift degrees across target words')
    ax.grid()
    # ax.legend(loc='best')
    fig.savefig('degree_distribution.png', dpi=300)
    plt.close()

    if len(sys.argv) > 2:
        corpora_dir = sys.argv[2]
        for language in sorted(data, reverse=True):
            print('Counting frequencies for %s...' % language)
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
                ipm = words[el] / (size / 1000000)
                data[language[el]].append(ipm)

        frequencies = []
        for language in sorted(data, reverse=True):
            freqs = [el[2] for el in data[language]]
            frequencies.append(freqs)
        fig, ax = plt.subplots()

        ax.boxplot(frequencies, labels=sorted(data, reverse=True), whis='range')
        plot_title = 'Word frequencies'
        ax.title(plot_title)
        ax.grid()
        ax.xlabel('')
        ax.ylabel('Istances per million (IPM)')
        ax.savefig('ipm.png', dpi=300)
        ax.close()
        ax.legend(loc='best')
        fig.savefig('degree_distribution.png', dpi=300)
