# python3
# coding: utf-8

import numpy as np
import argparse
import logging
from sklearn.decomposition import PCA
from sklearn import preprocessing

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to 1st npz file with the embeddings', required=True)
    arg('--input1', '-i1', help='Path to 2nd npz file with the embeddings', required=True)
    arg('--target', '-t', help='Path to target words', required=True)
    parser.add_argument('--mode', '-m', default='mean', choices=['mean', 'pca'])

    args = parser.parse_args()
    data_path0 = args.input0
    data_path1 = args.input1

    target_words = set([w.strip() for w in open(args.target, 'r').readlines()])

    array0 = np.load(data_path0)
    logger.info('Loaded an array of %d entries from %s' % (len(array0), data_path0))

    array1 = np.load(data_path1)
    logger.info('Loaded an array of %d entries from %s' % (len(array1), data_path1))

    for word in target_words:
        if array0[word].shape[0] < 3 or array1[word].shape[0] < 3:
            logger.info('%s omitted because of low frequency' % word)
            print('\t'.join([word.split('_')[0], '10']))
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
        else:
            for m in [vectors0, vectors1]:
                vector = np.average(m, axis=0)
                vectors.append(vector)
        vectors = [preprocessing.normalize(v.reshape(1, -1), norm='l2') for v in vectors]
        shift = 1 / np.dot(vectors[0], vectors[1])
        print('\t'.join([word.split('_')[0], str(shift)]))
