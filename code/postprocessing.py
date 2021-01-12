import argparse
import logging
import pickle
import stanza
import time
import numpy as np
from collections import defaultdict
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
        '--frequency_type', type=str,
        help='Whether to correct for word frequency using log frequency ("log") or zipf frequency ("zipf"). '
             'If blank, no frequency correction is performed.')
    parser.add_argument(
        '--frequency_list', type=str,
        help='Path to a frequency list tsv file (word\tfreq[\trank]\n) to use instead of the wordfreq library. '
             'Only usable with log frequency as frequency_type.')
    args = parser.parse_args()

    lang = args.lang.lower()
    assert lang in ['en', 'de', 'sv', 'la', 'ru', 'it']
    assert args.frequency_type in [None, 'log', 'zipf']
    if args.frequency_list and args.frequency_type == 'zipf':
        raise NotImplementedError('No Zipf frequencies available with custom frequency list.')

    with open(args.subs_path, 'rb') as f_in:
        substitutes_pre = pickle.load(f_in)

    start_time = time.time()

    for lemma in substitutes_pre:
        for occurrence in substitutes_pre[lemma]:

            # log p(c_j|w,s_i) = log p(c_j|s_i) + log p(c_j|w), with p(c_j|w) = exp(dot(emb_c_j, embed_w))
            for i, dotp in enumerate(occurrence['dot_products']):
                occurrence['logp'][i] += dotp / args.temperature

            # sort candidates by p(c_j|w,s_i)
            indices = np.argsort(occurrence['logp'])[::-1]
            occurrence['logp'] = [occurrence['logp'][j] for j in indices]
            occurrence['candidates'] = [occurrence['candidates'][j] for j in indices]

    if args.frequency_type:
        logger.warning('Correct for word frequency.')
        if args.frequency_list:
            logger.warning('Loading frequency list.')
            freqs_tmp = dict()
            with open(args.frequency_list, 'r') as f_in:
                for line in f_in:
                    line = line.strip('\n').strip()
                    w, fr = line.split('\t')[:2]
                    freqs_tmp[w] = int(fr)

            sum_fr = sum(freqs_tmp.values())
            for w in freqs_tmp:
                freqs_tmp[w] = np.log(freqs_tmp[w] / sum_fr)

            log_relative_freqs = defaultdict(lambda: min(freqs_tmp.values()))
            log_relative_freqs.update(freqs_tmp)

        for target in substitutes_pre:
            for occurrence in substitutes_pre[target]:
                for w, logp in zip(occurrence['candidates'], occurrence['logp']):
                    if args.frequency_list:
                        logp -= log_relative_freqs[w]
                    elif args.frequency_type == 'zipf':
                        logp -= zipf_frequency(w, lang, wordlist='best')
                    else:
                        logp -= np.log(word_frequency(w, lang, wordlist='best'))

    if args.lemmatise:
        logger.warning('Lemmatisation postprocessing.')
        try:
            nlp = stanza.Pipeline(lang=lang, processors='tokenize, lemma')
        except FileNotFoundError:
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
                        substitutes_post[target][i]['logp'][subs_lemmas[sub_lemma]] = np.log(p)
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
            occurrence['logp'] -= np.log(np.sum(np.exp(occurrence['logp'])))

    with open(args.output_path, 'wb') as f_out:
        pickle.dump(substitutes_post, f_out)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
