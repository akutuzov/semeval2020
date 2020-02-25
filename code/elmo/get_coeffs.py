# python3
# coding: utf-8

import argparse
from elmo_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to 1st npz file with the embeddings', required=True)
    arg('--input1', '-i1', help='Path to 2nd npz file with the embeddings', required=True)
    parser.add_argument('--mode', '-m', default='centroid', choices=['centroid', 'pairwise'])

    args = parser.parse_args()
    data_path0 = args.input0
    data_path1 = args.input1

    coeffs0 = calc_coeffs(data_path0)
    coeffs1 = calc_coeffs(data_path1)

    assert len(coeffs0) == len(coeffs1)

    for word in coeffs0:
        coeff0 = coeffs0[word]
        coeff1 = coeffs1[word]
        delta = abs(coeff0 - coeff1)
        print('\t'.join([word, str(coeff0), str(coeff1), str(delta)]))

