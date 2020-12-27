import logging
import pickle
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfTransformer
import time
import numpy as np
from docopt import docopt
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

    # Get the arguments
    args = docopt("""Word sense induction via agglomerative clustering of lexical substitutes.
    
    Input format (<subsPath>): pickle file containing a dictionary. Keys are target words. 
    Values are lists with as many elements as target word occurrences. A list element is a 
    dictionary containing the ranked candidate tokens (key 'candidates') and the ranked log
    probabilities (key 'logp').

    Usage:
        wsi.py [--tfidf --nClusters=K --affinity=A --linkage=L] <subsPathT1> <subsPathT2> <outPath>

    Arguments:
        <subsPathT1> = path to pickle containing substitute lists for period 1
        <subsPathT2> = path to pickle containing substitute lists for period 2
        <outPath> = output path for tsv file containing JSD scores
    Options:
        --tfidf  Whether to use tf-idf before clustering
        --nClusters=K  The number of clusters to find  
        --affinity=A  Metric used to compute the linkage [default: cosine]
        --linkage=L  Which linkage criterion to use [default: average]
    """)

    subsPathT1 = args['<subsPathT1>']
    subsPathT2 = args['<subsPathT2>']
    outPath = args['<outPath>']
    useTfidf = bool(args['--tfidf'])
    nClusters = int(args['--nClusters'])
    affinity = args['--affinity']
    linkage = args['--linkage']

    with open(subsPathT1, 'rb') as f_in:
        substitutes_t1 = pickle.load(f_in)
    with open(subsPathT2, 'rb') as f_in:
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

        if useTfidf:
            logger.warning('Apply tf-idf.')
            tfidf = TfidfTransformer()
            m = tfidf.fit_transform(m).toarray()

        logger.warning('Cluster into {} cluster.'.format(nClusters))
        clustering = AgglomerativeClustering(
            n_clusters=nClusters,
            affinity=affinity,
            linkage=linkage
        )
        labels = clustering.fit_predict(m)

        logger.warning('Compute JSD.')
        usage_distr_t1 = np.zeros(nClusters)
        usage_distr_t2 = np.zeros(nClusters)
        for j, label in enumerate(labels):
            if j < n_occ_t1:
                usage_distr_t1[label] += 1
            else:
                usage_distr_t2[label] += 1

        usage_distr_t1 /= usage_distr_t1.sum()
        usage_distr_t2 /= usage_distr_t2.sum()

        jsd_scores[target] = jsd(usage_distr_t1, usage_distr_t2)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))

    with open(outPath, 'w', encoding='utf-8') as f:
        for word, score in jsd_scores.items():
            f.write("{}\t{}\n".format(word, score))


if __name__ == '__main__':
    main()
