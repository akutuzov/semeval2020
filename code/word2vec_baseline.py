# python3
# coding: utf-8

import gensim
import logging
import argparse
from smart_open import open
import numpy as np
from os import path
from procrustes import smart_procrustes_align_gensim

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--emb_dir', '-e', help='Directory with embeddings', required=True)
    arg('--targets', '-t', help='File with target words', required=True)
    arg('--pos', '-p', help='Add POS tags to words?', default='no', choices=['no', 'yes'])
    arg('--mode', '-m', default='incremental', choices=['incremental', 'align'])

    args = parser.parse_args()

    targets = set([w.strip() for w in open(args.targets, 'r').readlines()])

    if args.mode == 'incremental':
        modelfiles = ['corpus1.model', 'corpus2_incremental.model']
    else:
        modelfiles = ['corpus1.model', 'corpus2.model']

    assert len(modelfiles) == 2

    models = []

    for mfile in modelfiles:
        model = gensim.models.KeyedVectors.load(path.join(args.emb_dir, mfile))
        model.init_sims(replace=True)
        models.append(model)

    similarities = {t: 0 for t in targets}

    if args.mode == 'align':
        logger.info('Aligning models...')
        models[1] = smart_procrustes_align_gensim(models[0], models[1])
        logger.info('Aligning complete...')

    for word in similarities:
        if args.pos == 'yes':
            word_n = word + '_NOUN'
            word_a = word + '_ADJ'
            a_freq = 0
            if word_a in models[0] and word_a in models[1]:
                word = word_a
                a_freq = models[0].vocab[word].count
            if word_n in models[0] and word_n in models[1]:
                n_freq = models[0].vocab[word_n].count
                if n_freq > a_freq:
                    word = word_n
        if word not in models[0] or word not in models[1]:
            logger.info('%s not found!' % word)
            shift = 10.0
        else:
            vector0 = models[0][word]
            vector1 = models[1][word]
            shift = 1 / np.dot(vector0, vector1)
        print('\t'.join([word.split('_')[0], str(shift)]))
