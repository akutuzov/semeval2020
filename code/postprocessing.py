import logging
import pickle
import stanza
import time
import numpy as np
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
        postprocessing.py [--nSubs=N --language=L --frequency=F --lemmatise] <subsPath> <outPath>
        
    Arguments:
        <subsPath> = path to pickle containing substitute lists
        <outPath> = output path for substitutes with updated probabilities
    Options:
        --nSubs=N  The number of lexical substitutes to keep 
        --language=L  The language code for word frequencies
        --frequency=F  Whether to correct for word frequency: 'log' or 'zipf'
        --lemmatise  Whether to lemmatise lexical substitutes
 
    """)

    subsPath = args['<subsPath>']
    outPath = args['<outPath>']
    nSubs = int(args['--nSubs']) if args['--nSubs'] else None
    lang = args['--language']
    correctFreq = args['--frequency']
    lemmatise = bool(args['--lemmatise'])

    assert lang.lower() in ['en', 'de', 'sw', 'la', 'ru', 'it']
    assert correctFreq in [None, 'log', 'zipf']

    with open(subsPath, 'rb') as f_in:
        substitutes_pre = pickle.load(f_in)

    start_time = time.time()

    if correctFreq:
        logger.warning('Correct for word frequency.')
        if lang == 'la':
            raise NotImplementedError('No Latin word frequencies available.')

        for target in substitutes_pre:
            for occurrence in substitutes_pre[target]:
                for w, logp in zip(occurrence['candidates'], occurrence['logp']):
                    if correctFreq == 'zipf':
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

        for target in substitutes_pre:
            tgt_lemma = nlp(target).sentences[0].words[0].lemma
            for occurrence in substitutes_pre[target]:
                subs_lemmas = {}

                for j, w in enumerate(occurrence['candidates']):
                    lemma = nlp(w).sentences[0].words[0].lemma

                    if lemma == tgt_lemma:
                        occurrence['logp'][j] = 100  # remove
                        continue

                    if lemma in subs_lemmas:
                        p = np.exp(occurrence['logp'][subs_lemmas[lemma]]) + np.exp(occurrence['logp'][j])
                        occurrence['logp'][subs_lemmas[lemma]] = np.log(p)
                        occurrence['logp'][j] = 100  # remove
                    else:
                        subs_lemmas[lemma] = j

    substitutes_post = {
        w: [{'candidates': [], 'logp': []} for _ in substitutes_pre[w]]
        for w in substitutes_pre
    }
    for target in substitutes_pre:
        for j, occurrence in enumerate(substitutes_pre[target]):
            for w, logp in zip(occurrence['candidates'], occurrence['logp']):
                if logp != 100:
                    substitutes_post[target][j]['candidates'].append(w)
                    substitutes_post[target][j]['logp'].append(logp)

    for lemma in substitutes_post:
        for occurrence in substitutes_post[lemma]:
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
