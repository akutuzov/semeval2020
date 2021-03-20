from collections import defaultdict
from gensim.models.word2vec import PathLineSentences
from docopt import docopt
import logging
import time
from smart_open import open


def main():
    """
    Get frequencies from corpus.
    """

    # Get the arguments
    args = docopt("""Get frequencies from corpus.

    Usage:
        freq.py [-n] <corpDir> <outPath>

    Arguments:
        <corpDir> = path to corpus or corpus directory (iterates through files)
        <outPath> = output path for result file

    Options:
        -n --norm  normalize frequency by total corpus frequency

    """)

    is_norm = args['--norm']
    corpDir = args['<corpDir>']
    outPath = args['<outPath>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # Get sentence iterator
    sentences = PathLineSentences(corpDir)

    # Initialize frequency dictionary
    freqs = defaultdict(int)

    # Iterate over sentences and words
    corpusSize = 0
    for sentence in sentences:
        for word in sentence:
            corpusSize += 1
            freqs[word] = freqs[word] + 1

    freqs = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=True))

    # Write frequency scores
    with open(outPath, 'w', encoding='utf-8') as f_out:
        for i, word in enumerate(freqs):
            if is_norm:
                freqs[word] = float(freqs[word]) / corpusSize  # Normalize by total corpus frequency
                f_out.write('\t'.join((word, str(freqs[word]), str(i))) + '\n')
            else:
                f_out.write('\t'.join((word, str(freqs[word]), str(i))) + '\n')



    logging.info('tokens: %d' % (corpusSize))
    logging.info('types: %d' % (len(freqs.keys())))
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
