import logging
import pickle
import stanza
import time
import numpy as np
from collections import defaultdict
from docopt import docopt
from wordfreq import word_frequency, zipf_frequency


logger = logging.getLogger(__name__)


def main():
    """
    Postprocessing: correct for substitute word frequency and lemmatise.
    """

    # Get the arguments
    args = docopt("""Postprocessing: correct for substitute word frequency and lemmatise.
    
    Input format (<subsPath>): pickle file containing a dictionary. Keys are target words. 
    Values are lists with as many elements as target word occurrences. A list element is a 
    dictionary containing the ranked candidate tokens (key 'candidates') and the ranked log
    probabilities (key 'logp').
    
    Usage:
        postprocessing.py [--nSubs=N --language=L --frequency=F --lemmatise --frequencyList=P] <subsPath> <outPath>
        
    Arguments:
        <subsPath> = path to pickle containing substitute lists
        <outPath> = output path for substitutes with updated probabilities
    Options:
        --nSubs=N  The number of lexical substitutes to keep 
        --language=L  The language code for word frequencies and lemmatisation
        --frequency=F  Whether to correct for word frequency: 'log' or 'zipf'
        --lemmatise  Whether to lemmatise lexical substitutes
        --frequencyList=P  Path to a frequency list tsv file (word\tfreq[\trank]\n)
    """)

    subsPath = args['<subsPath>']
    outPath = args['<outPath>']
    nSubs = int(args['--nSubs']) if args['--nSubs'] else None
    lang = args['--language']
    correctFreq = args['--frequency']
    lemmatise = bool(args['--lemmatise'])
    frequencyList = args['--frequencyList']

    lang = lang.lower()
    assert lang in ['en', 'de', 'sw', 'la', 'ru', 'it']
    assert correctFreq in [None, 'log', 'zipf']
    if frequencyList and correctFreq == 'zipf':
        raise NotImplementedError('No Zipf frequencies available with custom frequency list.')

    with open(subsPath, 'rb') as f_in:
        substitutes_pre = pickle.load(f_in)

    start_time = time.time()

    if correctFreq:
        logger.warning('Correct for word frequency.')
        if frequencyList:
            logger.warning('Loading frequency list.')
            freqs_tmp = dict()
            with open(frequencyList, 'r') as f_in:
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
                    if frequencyList:
                        logp -= log_relative_freqs[w]
                    elif correctFreq == 'zipf':
                        logp -= zipf_frequency(w, lang, wordlist='best')
                    else:
                        logp -= np.log(word_frequency(w, lang, wordlist='best'))

    if lemmatise:
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
            if nSubs:
                occurrence['logp'] = occurrence['logp'][:nSubs]
                occurrence['candidates'] = occurrence['candidates'][:nSubs]

            # re-normalise
            occurrence['logp'] -= np.log(np.sum(np.exp(occurrence['logp'])))

    with open(outPath, 'wb') as f_out:
        pickle.dump(substitutes_post, f_out)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
