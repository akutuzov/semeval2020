# python3
# coding: utf-8

import numpy as np
import argparse
import logging
from sklearn.decomposition import PCA
from sklearn import preprocessing

# Cosine similarity (COS) algorithm

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to 1st npz file with the embeddings', required=True)
    arg('--input1', '-i1', help='Path to 2nd npz file with the embeddings', required=True)
    arg('--target', '-t', help='Path to target words', required=True)
    arg('--output', '-o', help='Output path (csv)', required=False)
    arg('--mode', '-m', default='mean', choices=['mean', 'pca', 'sum'])
    arg('-f', action='store_true', help='Output frequencies?')

    args = parser.parse_args()
    data_path0 = args.input0
    data_path1 = args.input1

    target_words = set([w.strip() for w in open(args.target, 'r', encoding='utf-8').readlines()])

    array0 = np.load(data_path0)
    logger.info('Loaded an array of {0} entries from {1}'.format(len(array0), data_path0))

    array1 = np.load(data_path1)
    logger.info('Loaded an array of {0} entries from {1}'.format(len(array1), data_path1))

    try:
        f_out = open(args.output, 'w', encoding='utf-8')
    except TypeError:
        f_out = None

    for word in target_words:
        frequency = np.median([array0[word].shape[0], array1[word].shape[0]])
        if array0[word].shape[0] < 3 or array1[word].shape[0] < 3:
            logger.info('{} omitted because of low frequency'.format(word))
            if args.f:
                print('\t'.join([word, '10', str(frequency)]), file=f_out)
            else:
                print('\t'.join([word, '10']), file=f_out)
            continue
        vectors0 = array0[word]
        vectors1 = array1[word]
        vectors = []
        if args.mode == 'pca':
            for m in [vectors0, vectors1]:
                scaled = (m - np.mean(m, 0)) / np.std(m, 0)
                pca = PCA(n_components=3)
                analysis = pca.fit(scaled)
                vector = analysis.components_[0]
                vectors.append(vector)
        elif args.mode == 'mean':
            for m in [vectors0, vectors1]:
                vector = np.average(m, axis=0)
                vectors.append(vector)
        elif args.mode == 'sum':
            for m in [vectors0, vectors1]:
                vector = np.sum(m, axis=0)
                vectors.append(vector)
        vectors = [preprocessing.normalize(v.reshape(1, -1), norm='l2') for v in vectors]
        shift = 1 / np.dot(vectors[0].reshape(-1), vectors[1].reshape(-1))
        if args.f:
            print('\t'.join([word, str(shift), str(frequency)]), file=f_out)
        else:
            print('\t'.join([word, str(shift)]), file=f_out)

    if f_out:
        f_out.close()
