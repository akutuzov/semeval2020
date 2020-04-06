# /usr/bin/env python3
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
from smart_open import open

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
        x_data = [float(el[1]) for el in data[language]]
        y_data = [language for el in x_data]
        ax.scatter(x_data, y_data, label=language)

        much_shifted = [val for val in x_data if val > 0.6]
        much_shifted_ratio = len(much_shifted) / len(x_data)
        print('%s: %0.3f of words with change degree > 0.6' % (language, much_shifted_ratio))

    ax.set(xlabel='Degree of change', ylabel='Languages',
           title='Distribution of shift degrees across target words')
    ax.grid()
    # ax.legend(loc='best')
    fig.savefig('degree_distribution.png', dpi=300)
    # plt.show()
