from collections import defaultdict
from docopt import docopt
import logging
import time
from scipy.sparse import dok_matrix, csr_matrix, save_npz
from utils_ import Space
from gensim.models.word2vec import PathLineSentences


def main():
    """
    Make count-based vector space from corpus.
    """

    # Get the arguments
    args = docopt("""Make count-based vector space from corpus.

    Usage:
        cnt.py <corpDir> <outPath> <windowSize>
        
    Arguments:
       
        <corpDir> = path to corpus or corpus directory (iterates through files)
        <outPath> = output path for vectors
        <windowSize> = the linear distance of context words to consider in each direction

    """)
    
    corpDir = args['<corpDir>']
    outPath = args['<outPath>']
    windowSize = int(args['<windowSize>'])    
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # Build vocabulary
    logging.info("Building vocabulary")
    sentences = PathLineSentences(corpDir)
    vocabulary = list(set([word for sentence in sentences for word in sentence if len(sentence)>1])) # Skip one-word sentences to avoid zero-vectors
    w2i = {w: i for i, w in enumerate(vocabulary)}
    
    # Initialize co-occurrence matrix as dictionary
    cooc_mat = defaultdict(lambda: 0)

    # Get counts from corpus
    sentences = PathLineSentences(corpDir)
    logging.info("Counting context words")
    for sentence in sentences:
        for i, word in enumerate(sentence):
            lowerWindowSize = max(i-windowSize, 0)
            upperWindowSize = min(i+windowSize, len(sentence))
            window = sentence[lowerWindowSize:i] + sentence[i+1:upperWindowSize+1]
            if len(window)==0: # Skip one-word sentences
                continue
            windex = w2i[word]
            for contextWord in window:
                cooc_mat[(windex,w2i[contextWord])] += 1

    
    # Convert dictionary to sparse matrix
    logging.info("Converting dictionary to matrix")
    cooc_mat_sparse = dok_matrix((len(vocabulary),len(vocabulary)), dtype=float)
    try:
        cooc_mat_sparse.update(cooc_mat)
    except NotImplementedError:
        cooc_mat_sparse._update(cooc_mat)
        
    # Save matrix
    Space(matrix=cooc_mat_sparse, rows=vocabulary, columns=vocabulary).save(outPath)   
        
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    
if __name__ == '__main__':
    main()
