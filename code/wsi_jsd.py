import argparse
import json
import logging
import os
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfTransformer
import time
import numpy as np
from scipy.stats import entropy
from smart_open import open


logger = logging.getLogger(__name__)


def jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def main():
    """
    Word sense induction using lexical substitutes.
    """
    parser = argparse.ArgumentParser(
        description='Word sense induction via agglomerative clustering of lexical substitutes.')
    parser.add_argument(
        '--subs_path_t1', type=str, required=True,
        help='Path to the pickle file containing substitute lists (output by postprocessing.py) for period T1.')
    parser.add_argument(
        '--subs_path_t2', type=str, required=True,
        help='Path to the pickle file containing substitute lists (output by postprocessing.py) for period T2.')
    parser.add_argument(
        '--targets_path', type=str, required=True,
        help='Path to the csv file containing target word forms.')
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='output path for csv file containing JSD scores')
    parser.add_argument(
        '--n_clusters', type=int, default=7,
        help='The number of clusters to find, fixed for all target words.')
    parser.add_argument(
        '--apply_tfidf', action='store_true',
        help="Whether to use tf-idf before clustering.")
    parser.add_argument(
        '--affinity', type=str, default='cosine',
        help='Metric used to compute the linkage.')
    parser.add_argument(
        '--linkage', type=str, default='average',
        help='linkage criterion to use.')
    args = parser.parse_args()

    if args.subs_path_t1.endswith('.pkl') and args.subs_path_t1.endswith('.pkl'):
        with open(args.subs_path_t1, 'rb') as f_in:
            substitutes_t1 = pickle.load(f_in)
        with open(args.subs_path_t2, 'rb') as f_in:
            substitutes_t2 = pickle.load(f_in)
    elif args.subs_path_t1.endswith('.json') and args.subs_path_t1.endswith('.json'):
        with open(args.subs_path_t1, 'r') as f_in:
            substitutes_t1 = json.load(f_in)
        with open(args.subs_path_t2, 'r') as f_in:
            substitutes_t2 = json.load(f_in)
    elif os.path.isdir(args.subs_path_t1) and os.path.isdir(args.subs_path_t2):
        substitutes_t1 = {}
        for fname in os.listdir(args.subs_path_t1):
            word = fname.split('_')[0]
            with open(os.path.join(args.subs_path_t1, fname), 'rb') as f_in:
                substitutes_t1[word] = [json.loads(jline) for jline in f_in.read().splitlines()]
        substitutes_t2 = {}
        for fname in os.listdir(args.subs_path_t2):
            word = fname.split('_')[0]
            with open(os.path.join(args.subs_path_t2, fname), 'rb') as f_in:
                substitutes_t2[word] = [json.loads(jline) for jline in f_in.read().splitlines()]

        if not substitutes_t1:
            logger.warning('No files in {} ?'.format(args.subs_path_t1))
        if not substitutes_t2:
            logger.warning('No files in {} ?'.format(args.subs_path_t2))

    else:
        raise ValueError('Invalid path: {}'.format(args.subs_path))


    # Load target forms
    targets = []
    with open(args.targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            targets.append(line.split(',')[0])
    print('=' * 80)
    print('')
    print('{} targets: {}'.format(len(targets), targets))
    print('=' * 80)

    start_time = time.time()

    # collect vocabulary of substitutes for all lemmas
    logger.warning('Collect vocabularies of substitutes.')

    vocabs = {target: set() for target in targets}
    n_occurrences = {target: 0 for target in targets}

    print(substitutes_t1['pin'])
    for target in targets:
        try:
            for occurrence in substitutes_t1[target]:
                vocabs[target] |= set(occurrence['candidate_words'])
                n_occurrences[target] += 1
        except KeyError:
            logger.warning('No occurrences of {} in T1.'.format(target))

        try:
            for occurrence in substitutes_t2[target]:
                vocabs[target] |= set(occurrence['candidate_words'])
                n_occurrences[target] += 1
        except KeyError:
            logger.warning('No occurrences of {} in T2.'.format(target))

    logger.warning('\nCollected vocabularies for {} targets.'.format(len([n for n in n_occurrences.values() if n > 0])))
    logger.warning('Total occurrences: {}'.format(sum(n_occurrences.values())))
    logger.warning('Minimum vocabulary size: {}'.format(min([len(v) for v in vocabs.values()])))
    logger.warning('Maximum vocabulary size: {}\n'.format(max([len(v) for v in vocabs.values()])))

    jsd_scores = {}
    for target in targets:
        logger.warning('Process "{}"'.format(target))

        if len(vocabs[target]) == 0:
            jsd_scores[target] = 1.
            logger.warning('Assigning JSD=1 to target word: {}. No substitutes available.'.format(target))
            continue

        # for each target, construct a one-hot matrix M where
        # cell M[i,j] encodes whether substitute j is in the
        # list of substitutes generated for sentence i
        logger.warning('Construct matrix.')
        w2i = {w: i for i, w in enumerate(vocabs[target])}
        m = np.zeros((n_occurrences[target], len(vocabs[target])))

        occ_idx = 0
        for occurrence in substitutes_t1[target]:
            for sub in occurrence['candidate_words']:
                m[occ_idx, w2i[sub]] = 1
            occ_idx += 1
        n_occ_t1 = occ_idx

        for occurrence in substitutes_t2[target]:
            for sub in occurrence['candidate_words']:
                m[occ_idx, w2i[sub]] = 1
            occ_idx += 1

        assert occ_idx == n_occurrences[target]

        if args.apply_tfidf:
            logger.warning('Apply tf-idf.')
            tfidf = TfidfTransformer()
            m = tfidf.fit_transform(m).toarray()

        logger.warning('Cluster into {} cluster.'.format(args.n_clusters))
        clustering = AgglomerativeClustering(
            n_clusters=args.n_clusters,
            affinity=args.affinity,
            linkage=args.linkage
        )
        try:
            labels = clustering.fit_predict(m)
        except ValueError as e:
            jsd_scores[target] = 1.
            logger.warning('Assigning JSD=1 to target word: {}'.format(target))
            logger.warning(e)
            continue

        logger.warning('Compute JSD.')
        usage_distr_t1 = np.zeros(args.n_clusters)
        usage_distr_t2 = np.zeros(args.n_clusters)
        for j, label in enumerate(labels):
            if j < n_occ_t1:
                usage_distr_t1[label] += 1
            else:
                usage_distr_t2[label] += 1

        usage_distr_t1 /= usage_distr_t1.sum()
        usage_distr_t2 /= usage_distr_t2.sum()

        jsd_scores[target] = jsd(usage_distr_t1, usage_distr_t2)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for word, score in jsd_scores.items():
            f.write("{}\t{}\n".format(word, score))


if __name__ == '__main__':
    main()
