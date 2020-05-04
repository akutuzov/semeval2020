# /usr/bin/env python3
# coding: utf-8

from test_set_stats import *

if __name__ == '__main__':
    method = 'COS'
    testset_dir = sys.argv[1]
    testset_files = [f for f in os.listdir(testset_dir) if f.endswith(method + '.txt')]
    print(method)
    data = {}
    for f in testset_files:
        lang = f.split('_')[0]
        lines = [l.strip() for l in open(os.path.join(testset_dir, f), 'r').readlines()]
        data[lang] = []
        for l in lines:
            data[lang].append(l.split('\t'))

    fig = degree_plot(data, method + '_distribution.png', add_title=': %s' % method)
