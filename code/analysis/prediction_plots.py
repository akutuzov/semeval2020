# /usr/bin/env python3
# coding: utf-8

from test_set_stats import *

if __name__ == '__main__':
    testset_dir = sys.argv[1]
    if len(sys.argv) > 2:
        method = sys.argv[2]
    else:
        method = ''
    testset_files = [f for f in os.listdir(testset_dir) if f.endswith(method + '.txt')]
    print(method)
    data = {}
    for f in testset_files:
        lang = f.split('_')[0].split('.')[0]
        lines = [line.strip() for line in open(os.path.join(testset_dir, f), 'r').readlines()]
        data[lang] = []
        for line in lines:
            data[lang].append(line.split('\t'))

    if method:
        fig = degree_plot(data, method + '_distribution.png', add_title=': %s' % method)
    else:
        fig = degree_plot(data, 'degree_distribution.png')
