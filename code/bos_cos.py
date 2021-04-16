import argparse
import json
import logging
import os
import pickle
from scipy.spatial.distance import euclidean, cosine
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
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
        help='Path to the pickle file containing substitute lists '
             '(output by postprocessing.py) for period T1.')
    parser.add_argument(
        '--subs_path_t2', type=str, required=True,
        help='Path to the pickle file containing substitute lists '
             '(output by postprocessing.py) for period T2.')
    parser.add_argument(
        '--targets_path', type=str, required=True,
        help='Path to the csv file containing target word forms.')
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='output path for csv file containing JSD scores')
    parser.add_argument(
        '--apply_tfidf', action='store_true',
        help="Whether to use tf-idf before clustering.")
    parser.add_argument(
        '--use_idf', action='store_true')
    parser.add_argument(
        '--sublinear_tf', action='store_true')
    parser.add_argument(
        '--vocab_percent', type=float, default=1.,
        help='Maximum number of vocabulary features is vocab_percent * vocab_len'
    )
    parser.add_argument(
        '--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])

    args = parser.parse_args()

    if args.subs_path_t1.endswith('.pkl') and args.subs_path_t2.endswith('.pkl'):
        with open(args.subs_path_t1, 'rb') as f_in:
            substitutes_t1 = pickle.load(f_in)
        with open(args.subs_path_t2, 'rb') as f_in:
            substitutes_t2 = pickle.load(f_in)
    elif args.subs_path_t1.endswith('.json') and args.subs_path_t2.endswith('.json'):
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
        raise ValueError('Path(s) not valid: --subs_path_t1, --subs_path_t2.')

    # Load target forms
    targets = []
    with open(args.targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            targets.append(line.split(',')[0])
    logger.warning('=' * 80)
    logger.warning('')
    logger.warning('{} targets: {}'.format(len(targets), targets))
    logger.warning('=' * 80)

    start_time = time.time()

    # collect vocabulary of substitutes for all lemmas
    logger.warning('Collecting vocabularies of substitutes.')

    vocabs = {target: set() for target in targets}
    n_occurrences = {target: 0 for target in targets}

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

    logger.warning(f"Collected vocabularies for "
                   f"{len([n for n in n_occurrences.values() if n > 0])} targets.")
    logger.warning('Total occurrences: {}'.format(sum(n_occurrences.values())))
    logger.warning('Minimum vocabulary size: {}'.format(min([len(v) for v in vocabs.values()])))
    logger.warning('Maximum vocabulary size: {}\n'.format(max([len(v) for v in vocabs.values()])))

    cos_scores = {}
    for idx, target in enumerate(targets, start=1):
        logger.warning('{}/{} Process "{}"'.format(idx, len(targets), target))

        if len(vocabs[target]) == 0:
            cos_scores[target] = 1.
            logger.warning(f"Assigning COS=1 to target word: {target}. No substitutes available.")
            continue
        if target not in substitutes_t1 or target not in substitutes_t2:
            cos_scores[target] = 1.
            logger.warning(f"Assigning COS=1 to target word: {target}. "
                           f"No occurrences in at least one corpus")
            continue

        # for each target, construct a one-hot matrix M where
        # cell M[i,j] encodes whether substitute j is in the
        # list of substitutes generated for sentence i
        logger.warning('Constructing the matrix...')

        subs_corpus1 = []
        occ_idx = 0
        for occurrence in substitutes_t1[target]:
            subs_corpus1.extend(occurrence['candidate_words'])
            occ_idx += 1

        subs_corpus2 = []
        for occurrence in substitutes_t2[target]:
            subs_corpus2.extend(occurrence['candidate_words'])
            occ_idx += 1

        assert occ_idx == n_occurrences[target]

        subs_corpus = [' '.join(subs_corpus1), ' '.join(subs_corpus2)]

        vectorizer = CountVectorizer(lowercase=False, max_features=int(args.vocab_percent * len(vocabs[target])))
        m = vectorizer.fit_transform(subs_corpus)

        logger.warning('Vocab: {} / {}'.format(len(vectorizer.get_feature_names()), len(vocabs[target])))

        # Apply tf-idf to the entire matrix
        if args.apply_tfidf:
            logger.warning('Applying tf-idf...')
            tfidf = TfidfTransformer(sublinear_tf=args.sublinear_tf, use_idf=args.use_idf)
            m = tfidf.fit_transform(m).toarray()

        if args.distance == 'cosine':
            cos_scores[target] = cosine(m[0, :], m[1, :])
        else:
            cos_scores[target] = euclidean(m[0, :], m[1, :])

    logger.warning("--- %s seconds ---" % (time.time() - start_time))

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for word, score in cos_scores.items():
            f.write("{}\t{}\n".format(word, score))


if __name__ == '__main__':
    main()
