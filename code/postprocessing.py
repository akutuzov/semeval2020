import argparse
import logging
import pickle
import stanza
import time
import numpy as np
from collections import defaultdict
from decimal import Decimal
from wordfreq import word_frequency, zipf_frequency


logger = logging.getLogger(__name__)


def main():
    """
    Correct probabilities with lexical similarity scores, correct for substitute word frequency, lemmatise,
    and filter out redundant candidates.
    """
    parser = argparse.ArgumentParser(
        description='Correct probabilities with lexical similarity scores, correct for substitute word frequency, '
                    'lemmatise, and filter out redundant candidates.')
    parser.add_argument(
        '--subs_path', type=str, required=True,
        help='Path to the pickle file containing substitute lists (output by inject_word_similarity.py).')
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output path for pickle containing substitutes with updated log probabilities.')
    parser.add_argument(
        '--lang', type=str, required=True,
        help='The language code for word frequencies and lemmatisation (e.g., "en", "sv", "ru")')
    parser.add_argument(
        '--n_subs', type=int, default=100,
        help='The number of lexical substitutes to keep.')
    parser.add_argument(
        '--temperature', type=float, default=1,
        help='The temperature value for the lexical similarity calculation.')
    parser.add_argument(
        '--lemmatise', action='store_true',
        help="Whether to lemmatise lexical substitutes, filtering out candidates redundant")
    parser.add_argument(
        '--frequency_correction', action='store_true',
        help='Whether to correct for word frequency using prior word probability distribution.')
    parser.add_argument(
        '--k', type=float, default=4,
        help='The value of parameter k in the prior word probability distribution.')
    parser.add_argument(
        '--s', type = float, default = 1.05,
        help = 'The value of parameter s in the prior word probability distribution.')
    parser.add_argument(
        '--beta', type=float, default=2,
        help='The value of parameter beta.')
    parser.add_argument(
        '--frequency_list', type=str,
        help='Path to a frequency list tsv file (word\tfreq[\trank]\n) to use instead of the wordfreq library.')
    args = parser.parse_args()

    lang = args.lang.lower()
    assert lang in ['en', 'de', 'sv', 'la', 'ru', 'it']

    with open(args.subs_path, 'rb') as f_in:
        substitutes_pre = pickle.load(f_in)

    start_time = time.time()

    for lemma in substitutes_pre:
        for occurrence in substitutes_pre[lemma]:

            # log p(c_j|w,s_i) = log p(c_j|s_i) + log p(c_j|w), with p(c_j|w) = exp(dot(emb_c_j, embed_w))
            for i, dotp in enumerate(occurrence['dot_products']):
                occurrence['logp'][i] += dotp / args.temperature
                # occurrence['logp'][i] = Decimal(occurrence['logp'][i])

            # sort candidates by p(c_j|w,s_i)
            indices = np.argsort(occurrence['logp'])[::-1]
            occurrence['logp'] = [occurrence['logp'][j] for j in indices]
            occurrence['candidates'] = [occurrence['candidates'][j] for j in indices]

    if args.frequency_correction:
        logger.warning('Correct for word frequency.')
        if args.frequency_list:
            logger.warning('Loading frequency list.')
            freqs_tmp = dict()
            log_prior_tmp = dict()
            with open(args.frequency_list, 'r') as f_in:
                for rnk, line in enumerate(f_in, start=1):
                    line = line.strip('\n').strip()
                    w, fr = line.split('\t')[:2]
                    freqs_tmp[w] = int(fr)

                    log_prior_tmp[w] = - np.log(args.k + rnk) * args.s  # [-0.7, -5] approx

            log_prior = defaultdict(lambda: min(log_prior_tmp.values()))
            log_prior.update(log_prior_tmp)

            # sum_fr = sum(freqs_tmp.values())
            # for w in freqs_tmp:
            #     freqs_tmp[w] = np.log(freqs_tmp[w] / sum_fr)
            #
            # log_relative_freqs = defaultdict(lambda: min(freqs_tmp.values()))
            # log_relative_freqs.update(freqs_tmp)

        for target in substitutes_pre:
            for occurrence in substitutes_pre[target]:
                prior_prob = []
                for w, logp in zip(occurrence['candidates'], occurrence['logp']):

                    if args.frequency_list:
                        logp -= args.beta * log_prior[w]
                    else:
                        logp -= args.beta * np.log(word_frequency(w, lang, wordlist='best') ** args.s)


    if args.lemmatise:
        logger.warning('Lemmatisation postprocessing.')
        try:
            nlp = stanza.Pipeline(lang=lang, processors='tokenize, lemma')
        except Exception:
            stanza.download(lang=lang, processors='tokenize, lemma')
            nlp = stanza.Pipeline(lang=lang, processors='tokenize, lemma')

        substitutes_post = {
            w: [{'candidates': [], 'logp': []} for _ in substitutes_pre[w]]
            for w in substitutes_pre
        }

        for target in substitutes_pre:
            tgt_lemma = nlp(target).sentences[0].words[0].lemma
            for i, occurrence in enumerate(substitutes_pre[target]):
                subs_lemmas = {}

                j = 0
                for sub, sub_logp in zip(occurrence['candidates'], occurrence['logp']):
                    sub_lemma = nlp(sub).sentences[0].words[0].lemma

                    if len(sub_lemma) <= 1:
                        continue

                    if sub_lemma == tgt_lemma:
                        continue

                    if sub_lemma in subs_lemmas:
                        p = np.exp(occurrence['logp'][subs_lemmas[sub_lemma]]) + np.exp(sub_logp)
                        substitutes_post[target][i]['logp'][subs_lemmas[sub_lemma]] = p.ln()
                    else:
                        subs_lemmas[sub_lemma] = j
                        substitutes_post[target][i]['candidates'].append(sub_lemma)
                        substitutes_post[target][i]['logp'].append(sub_logp)
                        j += 1
    else:
        substitutes_post = substitutes_pre

    for sub_lemma in substitutes_post:
        for occurrence in substitutes_post[sub_lemma]:
            indices = np.argsort(occurrence['logp'])[::-1]
            occurrence['logp'] = [occurrence['logp'][j] for j in indices]
            occurrence['candidates'] = [occurrence['candidates'][j] for j in indices]
            if args.n_subs:
                occurrence['logp'] = occurrence['logp'][:args.n_subs]
                occurrence['candidates'] = occurrence['candidates'][:args.n_subs]

            # re-normalise
            log_denominator = np.sum(np.exp(occurrence['logp'])).ln()
            for logp in occurrence['logp']:
                logp -= log_denominator

    with open(args.output_path, 'wb') as f_out:
        pickle.dump(substitutes_post, f_out)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
