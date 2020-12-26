from wordfreq import word_frequency, zipf_frequency
import pickle
import spacy
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from docopt import docopt


logger = logging.getLogger(__name__)

def main():
    """
    Postprocessing: correct for substitute word frequency and lemmatise.
    """

    # Get the arguments
    args = docopt("""Collect BERT representations from corpus.

    Usage:
        postprocessing.py [--nSubs=N --frequency --language=L --lemmatise] <subsPath> <outPath>
        
    Arguments:
        <subsPath> = path to pickle containing substitute lists
        <outPath> = output path for substitutes with updated probabilities
    Options:
        --nSubs=N  The number of lexical substitutes to extract 
        --language=L  The language code for word frequencies
        --frequency  Whether to correct for word frequency
        --lemmatise  Whether to lemmatise lexical substitutes
 
    """)

    subsPath = args['<subsPath>']
    outPath = args['<outPath>']
    nSubs = int(args['--nSubs']) if args['--nSubs'] else None
    lang = args['--language']
    correctFreq = bool(args['--frequency'])
    lemmatise = bool(args['--lemmatise'])

    assert lang.lower() in ['en', 'de', 'sw', 'la', 'ru', 'it']

    with open(subsPath, 'rb') as f_in:
        substitutes_pre = pickle.load(f_in)

    start_time = time.time()

    if correctFreq:
        for target in substitutes_pre:
            for occurrence in substitutes_pre[target]:
                for w, logp in zip(occurrence['candidates'], occurrence['logp']):
                    logp -= zipf_frequency(w, lang, wordlist='best')
                    logp -= np.log(word_frequency(w, lang, wordlist='best'))

    if lemmatise:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger', 'tokenizer'])

        for target in substitutes_pre:
            tgt_lemma = nlp(target)[0].lemma_
            for occurrence in substitutes_pre[target]:
                subs_lemmas = {}

                for j, w in enumerate(occurrence['candidates']):
                    lemma = nlp(w)[0].lemma_

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
