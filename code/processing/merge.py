# python3
# coding: utf-8

import argparse
import sys
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to directory with npz files', required=True)
    arg('--output', '-o', help='Path to the output file', required=True)

    args = parser.parse_args()
    data_path = args.input
    output = args.output
    files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

    array = {}

    for f in files:
        print('Processing', f, file=sys.stderr)
        word = f.split('.')[0]
        cur_array = np.load(os.path.join(data_path, f))
        array[word] = cur_array['arr_0']

    print('Loaded an array of %d entries' % len(array), file=sys.stderr)

    np.savez_compressed(output, **array)

    print('Vectors saved to', output, file=sys.stderr)
