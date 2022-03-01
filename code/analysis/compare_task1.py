import argparse
from collections import defaultdict

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-g', '--gold', help='Path to target words with gold binary change scores.', required=True)
    arg('-p', '--predictions', nargs='+', help='One or more files with binary predictions.', required=True)
    arg('-o', '--output', help='File path for output tsv.', required=True)
    args = parser.parse_args()

    gold = {}
    with open(args.gold, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            lemma, score = line.split('\t')
            gold[lemma] = score
    gold = dict(sorted(gold.items(), key=lambda item: item[1], reverse=True))

    predictions = defaultdict(dict)
    methods = []
    for filepath in args.predictions:
        method_name = filepath.split('/')[-1]
        method_name = method_name.split('.')[0]
        methods.append(method_name)
        with open(filepath, 'r', encoding='utf-8') as f_in:
            for line in f_in.readlines():
                line = line.strip()
                lemma, score = line.split('\t')
                predictions[method_name][lemma] = score

    accuracies = defaultdict(list)
    with open(args.output, 'w', encoding='utf-8') as f_out:
        f_out.write('lemma\tgold\t{}\n'.format('\t'.join(methods)))
        for lemma in gold:
            all_predictions_lemma = [predictions[m][lemma] for m in methods]
            f_out.write('{}\t{}\t{}\n'.format(lemma, gold[lemma], '\t'.join(all_predictions_lemma)))
            for m in methods:
                accuracies[m].append(int(gold[lemma] == predictions[m][lemma]))

        f_out.write('\t\t{}\n'.format('\t'.join([str(np.mean(accuracies[m])) for m in methods])))


