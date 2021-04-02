import argparse
from collections import defaultdict
from gensim.models.word2vec import PathLineSentences
from docopt import docopt
import logging
import time
from smart_open import open


def main():

    parser = argparse.ArgumentParser(description='Get frequencies from corpus.')
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to corpus or corpus directory (iterates through files).'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output path for result file.'
    )
    parser.add_argument(
        '--norm', action='store_true',
        help='Normalize frequency by total corpus frequency.'
    )
    parser.add_argument(
        '--lower', action='store_true',
        help='Apply lowercasing to the corpus.'
    )
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # Get sentence iterator
    sentences = PathLineSentences(args.data_path)

    # Initialize frequency dictionary
    freqs = defaultdict(int)

    # Iterate over sentences and words
    corpusSize = 0
    for sentence in sentences:
        for word in sentence:
            corpusSize += 1
            if args.lower:
                freqs[word.lower()] = freqs[word.lower()] + 1
            else:
                freqs[word] = freqs[word] + 1

    freqs = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=True))

    # Write frequency scores
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for i, word in enumerate(freqs, start=1):
            if args.norm:
                freqs[word] = float(freqs[word]) / corpusSize  # Normalize by total corpus frequency
                f_out.write('\t'.join((word, str(freqs[word]), str(i))) + '\n')
            else:
                f_out.write('\t'.join((word, str(freqs[word]), str(i))) + '\n')

    logging.info('tokens: %d' % (corpusSize))
    logging.info('types: %d' % (len(freqs.keys())))
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
