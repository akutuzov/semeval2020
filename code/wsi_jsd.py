import argparse
import logging
import pickle
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfTransformer
import time
import numpy as np
from scipy.stats import entropy


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

    with open(args.subs_path_t1, 'rb') as f_in:
        substitutes_t1 = pickle.load(f_in)
    with open(args.subs_path_t2, 'rb') as f_in:
        substitutes_t2 = pickle.load(f_in)

    start_time = time.time()

    # collect vocabulary of substitutes for all lemmas
    logger.warning('Collect vocabularies of substitutes.')

    vocabs = defaultdict(set)
    n_occurrences = defaultdict(int)

    for target in substitutes_t1:
        for occurrence in substitutes_t1[target]:
            vocabs[target] |= set(occurrence['candidates'])
            n_occurrences[target] += 1

    for target in substitutes_t2:
        for occurrence in substitutes_t2[target]:
            vocabs[target] |= set(occurrence['candidates'])
            n_occurrences[target] += 1

    logger.warning('Collected vocabularies for {} targets.'.format(len(vocabs)))
    logger.warning('Total occurrences: {}'.format(sum(n_occurrences.values())))
    logger.warning('Minimum vocabulary size: {}'.format(min([len(v) for v in vocabs.values()])))
    logger.warning('Maximum vocabulary size: {}'.format(max([len(v) for v in vocabs.values()])))

    jsd_scores = {}
    for target in vocabs:
        logger.warning('Process "{}"'.format(target))

        # for each target, construct a one-hot matrix M where
        # cell M[i,j] encodes whether substitute j is in the
        # list of substitutes generated for sentence i
        logger.warning('Construct matrix.')
        w2i = {w: i for i, w in enumerate(vocabs[target])}
        m = np.zeros((n_occurrences[target], len(vocabs[target])))

        occ_idx = 0
        for occurrence in substitutes_t1[target]:
            for sub in occurrence['candidates']:
                m[occ_idx, w2i[sub]] = 1
            occ_idx += 1
        n_occ_t1 = occ_idx

        for occurrence in substitutes_t2[target]:
            for sub in occurrence['candidates']:
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
        labels = clustering.fit_predict(m)

        print(labels)

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

        print(usage_distr_t1, usage_distr_t2)

        jsd_scores[target] = jsd(usage_distr_t1, usage_distr_t2)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for word, score in jsd_scores.items():
            f.write("{},{}\n".format(word, score))


if __name__ == '__main__':
    main()
