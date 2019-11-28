from docopt import docopt
import logging
import time
from scipy.spatial.distance import cosine
from utils_ import Space


def main():
    """
    Compute cosine distance for targets in two matrices.
    """

    # Get the arguments
    args = docopt("""Compute cosine distance for targets in two matrices.

    Usage:
        cd.py <testset> <matrix1> <matrix2> <outPath>

        <testset> = path to file with one target per line
        <matrix1> = path to matrix1 in npz format
        <matrix2> = path to matrix2 in npz format
        <outPath> = output path for result file

     Note:
         Important: spaces must be already aligned (columns in same order)!
        
    """)
    
    matrix1 = args['<matrix1>']
    matrix2 = args['<matrix2>']
    testset = args['<testset>']
    outPath = args['<outPath>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()    
    
    # Load matrices and rows
    space1 = Space(path=matrix1)   
    space2 = Space(path=matrix2)   
    matrix1 = space1.matrix
    row2id1 = space1.row2id
    matrix2 = space2.matrix
    row2id2 = space2.row2id
    
    # Load targets
    with open(testset, 'r', encoding='utf-8') as f_in:
            targets = [line.strip() for line in f_in]
        
    scores = {}
    for target in targets:
        
        # Get row vectors
        try:
            v1 = matrix1[row2id1[target]].toarray().flatten()
            v2 = matrix2[row2id2[target]].toarray().flatten()
        except KeyError:
            scores[target] = 'nan'
            continue
        
        # Compute cosine distance of vectors
        distance = cosine(v1, v2)
        scores[target] = distance
        
        
    with open(outPath, 'w', encoding='utf-8') as f_out:
        for target in targets:
            f_out.write('\t'.join((target, str(scores[target])+'\n')))

                
    logging.info("--- %s seconds ---" % (time.time() - start_time))                   
    
    

if __name__ == '__main__':
    main()
