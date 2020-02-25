# python3
# coding: utf-8

import argparse
import sys
from elmo_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to an npz file with the embeddings', required=True)
    parser.add_argument('--mode', '-m', default='centroid', choices=['centroid', 'pairwise'])

    args = parser.parse_args()
    data_path = args.input

    array = np.load(data_path)

    print('Loaded an array of %d entries' % len(array), file=sys.stderr)

    for word in array:
        if array[word].shape[0] < 3:
            print(word, 'omitted because of low frequency:', array[word].shape[0], file=sys.stderr)
            continue
        if args.mode == 'pairwise':
            var_coeff = pairwise_diversity(array[word])
        else:
            var_coeff = diversity(array[word])
        print(word+'\t', var_coeff)

    print('Variation coefficients produced', file=sys.stderr)
