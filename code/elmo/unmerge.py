# python3
# coding: utf-8

import argparse
import sys
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to the npz file', required=True)
    arg('--output', '-o', help='Path to the output directory', required=True)

    args = parser.parse_args()
    data_path = args.input
    output = args.output

    array = np.load(data_path)
    print('Loaded an array of %d entries' % len(array.files), file=sys.stderr)

    for f in array.files:
        print('Processing', f, file=sys.stderr)
        np.savez_compressed(os.path.join(output, f),array[f])

    print('Vectors saved to', output, file=sys.stderr)
