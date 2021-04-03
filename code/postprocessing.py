import argparse
import json
import logging
import numpy as np
import os
import pickle
import stanza
import time
from collections import defaultdict
from smart_open import open
from tqdm import tqdm
from wordfreq import word_frequency
from ufal.udpipe import Model, Pipeline

logger = logging.getLogger(__name__)


def lemm_udpipe(pipeline, word):
    # lemmatizing and processing the resulting CONLLU
    processed = pipeline.process(word)
    content = [line for line in processed.split("\n") if not line.startswith("#")]
    tagged = [w.split("\t") for w in content if w]
    (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = tagged[0]
    return lemma


def main():
    """
    Correct probabilities with lexical similarity scores, correct for substitute word frequency,
    lemmatise, and filter out redundant candidates.
    """
    parser = argparse.ArgumentParser(
        description='Correct probabilities with lexical similarity scores, '
                    'correct for substitute word frequency, '
                    'lemmatise, and filter out redundant candidates.')
    parser.add_argument(
        '--subs_path', type=str, required=True,
        help='Path to the pickle file containing substitute lists (output by inject_word_similarity.py).'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output path for pickle containing substitutes with updated log probabilities.'
    )
    parser.add_argument(
        "--output_postfix", default="_substitutes_post.json.gz",
        help="Out file postfix (added to the word)"
    )
    parser.add_argument(
        '--lang', type=str, required=True, choices=['en', 'sv', 'la', 'de', 'ru'],
        help='The language code for word frequencies and lemmatisation.'
    )
    parser.add_argument(
        '--n_subs', type=int, default=100,
        help='The number of lexical substitutes to keep.'
    )
    parser.add_argument(
        '--temperature', type=float, default=1,
        help='The temperature value for the lexical similarity calculation.'
    )
    parser.add_argument(
        '--lemmatise', action='store_true',
        help="Whether to lemmatise lexical substitutes, filtering out candidates redundant"
    )
    parser.add_argument(
        '--frequency_correction', action='store_true',
        help='Whether to correct for word frequency using prior word probability distribution.'
    )
    parser.add_argument(
        '--k', type=float, default=4,
        help='The value of parameter k in the prior word probability distribution.'
    )
    parser.add_argument(
        '--s', type=float, default=1.05,
        help='The value of parameter s in the prior word probability distribution.'
    )
    parser.add_argument(
        '--beta', type=float, default=2,
        help='The value of parameter beta.'
    )
    parser.add_argument(
        '--frequency_list', type=str,
        help='Path to a frequency-rank tsv file (word\tfreq[\trank]\n).'
    )
    parser.add_argument(
        '--lemmatizer', default="udpipe", choices=["stanza", "udpipe"],
        help='The lemmatizer to use.'
    )
    parser.add_argument(
        '--udfile', default="english-lines-ud-2.5-191206.udpipe",
        help='UDPipe model to use: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131'
    )
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        assert os.path.isdir(args.output_path), 'Output path must be a directory.'
    else:
        os.makedirs(args.output_path)

    if args.subs_path.endswith('.pkl'):
        with open(args.subs_path, 'rb') as f_in:
            substitutes_pre = pickle.load(f_in)
    elif args.subs_path.endswith('.json'):
        with open(args.subs_path, 'r') as f_in:
            substitutes_pre = json.load(f_in)
    elif os.path.isdir(args.subs_path):
        substitutes_pre = {}
        for fname in os.listdir(args.subs_path):
            word = fname.split('_')[0]
            with open(os.path.join(args.subs_path, fname), 'rb') as f:
                substitutes_pre[word] = [json.loads(jline) for jline in f.read().splitlines()]
    else:
        raise ValueError('Invalid path: {}'.format(args.subs_path))


    start_time = time.time()

    for lemma in substitutes_pre:
        for occurrence in substitutes_pre[lemma]:

            # log p(c_j|w,s_i) = log p(c_j|s_i) + log p(c_j|w),
            # with p(c_j|w) = exp(dot(emb_c_j, embed_w))
            for i, dotp in enumerate(occurrence['dot_products']):
                occurrence['logp'][i] += dotp / args.temperature
                # occurrence['logp'][i] = Decimal(occurrence['logp'][i])

            # sort candidates by p(c_j|w,s_i)
            indices = np.argsort(occurrence['logp'])[::-1]
            occurrence['logp'] = [occurrence['logp'][j] for j in indices]
            occurrence['candidate_words'] = [occurrence['candidate_words'][j] for j in indices]

    if args.frequency_correction:
        logger.warning('Correct for word frequency.')
        log_prior = None
        if args.frequency_list:
            logger.warning('Loading frequency list.')
            log_prior_tmp = dict()
            with open(args.frequency_list, 'r') as f_in:
                for line in f_in:
                    line = line.strip('\n').strip()
                    w, fr, rnk = line.split('\t')
                    log_prior_tmp[w] = - np.log(args.k + int(rnk)) * args.s  # [-0.7, -5] approx

            log_prior = defaultdict(lambda: min(log_prior_tmp.values()))
            log_prior.update(log_prior_tmp)

        for target in substitutes_pre:
            for occurrence in substitutes_pre[target]:
                for w, logp in zip(occurrence['candidate_words'], occurrence['logp']):

                    if args.frequency_list:
                        logp -= args.beta * log_prior[w]
                    else:
                        logp -= args.beta * np.log(
                            word_frequency(w, args.lang, wordlist='best') ** args.s)

    if args.lemmatise:
        logger.warning('Lemmatisation postprocessing.')
        if args.lemmatizer == "udpipe":
            lemm_model = Model.load(args.udfile)
            nlp = Pipeline(lemm_model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        else:
            try:
                nlp = stanza.Pipeline(lang=args.lang, processors='tokenize, lemma')
            except Exception:
                stanza.download(lang=args.lang, processors='tokenize, lemma')
                nlp = stanza.Pipeline(lang=args.lang, processors='tokenize, lemma')

        substitutes_post = {
            w: [{'candidate_words': [], 'logp': []} for _ in substitutes_pre[w]]
            for w in substitutes_pre
        }

        for target in tqdm(substitutes_pre):
            if args.lemmatizer == "udpipe":
                tgt_lemma = lemm_udpipe(nlp, target)
            else:
                tgt_lemma = nlp(target).sentences[0].words[0].lemma
            for i, occurrence in enumerate(substitutes_pre[target]):
                subs_lemmas = {}

                j = 0
                for sub, sub_logp in zip(occurrence['candidate_words'], occurrence['logp']):
                    if args.lemmatizer == "udpipe":
                        sub_lemma = lemm_udpipe(nlp, sub)
                    else:
                        sub_lemma = nlp(sub).sentences[0].words[0].lemma

                    if len(sub_lemma) <= 1:
                        continue

                    if sub_lemma == tgt_lemma:
                        continue

                    if sub_lemma in subs_lemmas:
                        p = np.exp(occurrence['logp'][subs_lemmas[sub_lemma]]) + np.exp(sub_logp)
                        substitutes_post[target][i]['logp'][subs_lemmas[sub_lemma]] = np.log(p)
                    else:
                        subs_lemmas[sub_lemma] = j
                        substitutes_post[target][i]['candidate_words'].append(sub_lemma)
                        substitutes_post[target][i]['logp'].append(sub_logp)
                        j += 1
    else:
        substitutes_post = substitutes_pre

    for sub_lemma in substitutes_post:
        for occurrence in substitutes_post[sub_lemma]:
            indices = np.argsort(occurrence['logp'])[::-1]
            occurrence['logp'] = [occurrence['logp'][j] for j in indices]
            occurrence['candidate_words'] = [occurrence['candidate_words'][j] for j in indices]
            if args.n_subs:
                occurrence['logp'] = occurrence['logp'][:args.n_subs]
                occurrence['candidate_words'] = occurrence['candidate_words'][:args.n_subs]

            # re-normalise
            log_denominator = np.log(np.sum(np.exp(occurrence['logp'])))  # .ln()
            for logp in occurrence['logp']:
                logp -= log_denominator

    for target in substitutes_post:
        for occurrence in substitutes_post[target]:
            for logp in occurrence['logp']:
                logp = float(logp)

    for word in substitutes_post:
        if len(substitutes_post[word]) < 1:
            logger.warning(f"No occurrences found for {word}!")
            continue
        outfile = os.path.join(args.output_path, word + args.output_postfix)
        with open(outfile, "w") as f:
            for occurrence in substitutes_post[word]:
                out = json.dumps(occurrence, ensure_ascii=False)
                f.write(out + "\n")
        logger.info(f"Substitutes saved to {outfile}")

    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
