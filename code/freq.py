from collections import defaultdict
from gensim.models.word2vec import PathLineSentences
from docopt import docopt
import logging
import time


def main():
    """
    Get frequencies from corpus.
    """

    # Get the arguments
    args = docopt("""Get frequencies from corpus.

    Usage:
        freq.py [-n] <corpDir> <testset> <outPath>
        
    Arguments:
       
        <corpDir> = path to corpus or corpus directory (iterates through files)
        <testset> = path to file with one target per line
        <outPath> = output path for result file

    Options:
        -n --norm  normalize frequency by total corpus frequency
        
    """)
    
    is_norm = args['--norm']
    corpDir = args['<corpDir>']
    testset = args['<testset>']
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

    # Load targets
    with open(testset, 'r', encoding='utf-8') as f_in:
            targets = [line.strip() for line in f_in]

    # Write frequency scores
    with open(outPath, 'w', encoding='utf-8') as f_out:
        for word in targets:
            if word in freqs:
                if is_norm:
                    freqs[word]=float(freqs[word])/corpusSize # Normalize by total corpus frequency
                f_out.write('\t'.join((word, str(freqs[word])+'\n')))
            else:
                f_out.write('\t'.join((word, 'nan'+'\n')))

                
    logging.info('tokens: %d' % (corpusSize))
    logging.info('types: %d' % (len(freqs.keys())))
    logging.info("--- %s seconds ---" % (time.time() - start_time))                   
    

if __name__ == '__main__':
    main()
