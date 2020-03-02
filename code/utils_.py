import itertools
import os
import pickle
import logging
from scipy.sparse import csr_matrix, load_npz, save_npz
from gensim import utils as gensim_utils

logger = logging.getLogger(__name__)


class Space(object):
    """
    Load and save Space objects.
    """
        
    def __init__(self, path=None, matrix=csr_matrix([]), rows=[], columns=[]):
        """
        Can be either initialized (i) by providing a path, (ii) by providing a matrix, rows and columns, or (iii) by providing neither, then an empty instance is created
        `path` should be path to a matrix in npz format, expects rows and columns in same folder at '[path]_rows' and '[path]_columns'
        `rows` list with row names
        `columns` list with column names
        """
        
        if path!=None:
            # Load matrix
            matrix = load_npz(path)
            # Load rows
            with open(path + '_rows', 'rb') as f:
                rows = pickle.load(f)
            # Load columns
            with open(path + '_columns', 'rb') as f:
                columns = pickle.load(f)

        row2id = {r:i for i, r in enumerate(rows)}
        id2row = {i:r for i, r in enumerate(rows)}
        column2id = {c:i for i, c in enumerate(columns)}
        id2column = {i:c for i, c in enumerate(columns)}

        self.matrix = csr_matrix(matrix)
        self.rows = rows
        self.columns = columns
        self.row2id = row2id
        self.id2row = id2row
        self.column2id = column2id
        self.id2column = id2column      
        
    def save(self, path):
        """
        `path` saves matrix at path in npz format, saves rows and columns as pickled lists in same folder at '[path]_rows' and '[path]_columns'
        """
        
        # Save matrix
        with open(path, 'wb') as f:
            save_npz(f, self.matrix)    
        # Save rows
        with open(path + '_rows', 'wb') as f:
            pickle.dump(self.rows, f)
        # Save columns
        with open(path + '_columns', 'wb') as f:
            pickle.dump(self.columns, f)


class PathLineSentences(object):
    """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory
    in alphabetical order by filename.

    The directory must only contain files that can be read by :class:`gensim.models.word2vec.LineSentence`:
    .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.

    The format of files (either text, or compressed text files) in the path is one sentence = one line,
    with words already preprocessed and separated by whitespace.

    Warnings
    --------
    Does **not recurse** into subdirectories.

    """
    def __init__(self, source, limit=None, max_sentence_length=100000):
        """
        Parameters
        ----------
        source : str
            Path to the directory.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).

        """
        self.source = source
        self.limit = limit
        self.max_sentence_length = max_sentence_length

        if os.path.isfile(self.source):
            logger.debug('single file given as source, rather than a directory of files')
            logger.debug('consider using models.word2vec.LineSentence for a single file')
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            logger.info('reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')
        logger.info('files read into PathLineSentences:%s', '\n'.join(self.input_files))

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            logger.info('reading file %s', file_name)
            with gensim_utils.smart_open(file_name) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = gensim_utils.to_unicode(line, encoding='latin-1').split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length
