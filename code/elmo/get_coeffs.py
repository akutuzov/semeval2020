# python3
# coding: utf-8

import argparse
import sys
from elmo_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to directory with npz files', required=True)
    parser.add_argument('--mode', '-m', default='centroid', choices=['centroid', 'pairwise'])

    args = parser.parse_args()
    data_path = args.input
    files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

    array = {}

    for f in files:
        print('Processing', f, file=sys.stderr)
        word = f.split('.')[0]
        cur_array = np.load(os.path.join(data_path, f))
        array[word] = cur_array['arr_0']

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
