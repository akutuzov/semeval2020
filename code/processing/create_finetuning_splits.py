import argparse
import os
import numpy as np
from gensim import utils as gensim_utils
np.random.seed(42)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus1_path", type=str, required=True,
        help="Path to directory containing the .bz2, .gz, or text file of the first corpus."
    )
    parser.add_argument(
        "--corpus2_path", type=str, required=True,
        help="Path to directory containing the .bz2, .gz, or text file of the second corpus."
    )
    parser.add_argument(
        "--corpus3_path", type=str, required=False,
        help="Optional path to directory containing the .bz2, .gz, or text file of the third corpus."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to output directory for train and test split to be stored."
    )
    parser.add_argument(
        "--test_percent", type=float, required=True,
        help="Percentage of the union of the two corpora to be reserved for the test split."
    )
    args = parser.parse_args()

    corpus1_dir = [f for f in os.listdir(args.corpus1_path) if not f.startswith('.') and f.endswith('.txt.gz')]
    corpus2_dir = [f for f in os.listdir(args.corpus2_path) if not f.startswith('.') and f.endswith('.txt.gz')]
    if args.corpus3_path:
        corpus3_dir = [f for f in os.listdir(args.corpus3_path) if not f.startswith('.') and f.endswith('.txt.gz')]

    assert len(corpus1_dir) == 1
    assert len(corpus2_dir) == 1
    if args.corpus3_path:
        assert len(corpus3_dir) == 1

    with gensim_utils.file_or_filename(os.path.join(args.corpus1_path, corpus1_dir[0])) as f:
        corpus1 = [gensim_utils.to_unicode(line, encoding='utf-8') for line in f]

    with gensim_utils.file_or_filename(os.path.join(args.corpus2_path, corpus2_dir[0])) as f:
        corpus2 = [gensim_utils.to_unicode(line, encoding='utf-8') for line in f]

    if args.corpus3_path:
        with gensim_utils.file_or_filename(os.path.join(args.corpus3_path, corpus3_dir[0])) as f:
            corpus3 = [gensim_utils.to_unicode(line, encoding='utf-8') for line in f]

    total_length = len(corpus1) + len(corpus2)
    if args.corpus3_path:
        total_length += len(corpus3)

    test_length = round(total_length * args.test_percent)
    train_length = total_length - test_length

    print('Total number of sentences in the corpus: {}'.format(total_length))
    print('Number of test sentences: {}'.format(test_length))

    # test corpus has the same amount of corpus 1 and corpus 2 (and corpus 3) sentences
    if args.corpus3_path:
        test_indices_corpus1 = np.random.choice(np.arange(len(corpus1)), int(test_length / 3), replace=False)
        test_indices_corpus2 = np.random.choice(np.arange(len(corpus2)), int(test_length / 3), replace=False)
        test_indices_corpus3 = np.random.choice(np.arange(len(corpus3)), int(test_length / 3), replace=False)
    else:
        test_indices_corpus1 = np.random.choice(np.arange(len(corpus1)), int(test_length / 2), replace=False)
        test_indices_corpus2 = np.random.choice(np.arange(len(corpus2)), int(test_length / 2), replace=False)

    train_indices_corpus1 = np.delete(np.arange(len(corpus1)), test_indices_corpus1)
    train_indices_corpus2 = np.delete(np.arange(len(corpus2)), test_indices_corpus2)
    if args.corpus3_path:
        train_indices_corpus3 = np.delete(np.arange(len(corpus3)), test_indices_corpus3)

    np.random.shuffle(test_indices_corpus1)
    np.random.shuffle(test_indices_corpus2)
    np.random.shuffle(train_indices_corpus1)
    np.random.shuffle(train_indices_corpus2)
    if args.corpus3_path:
        np.random.shuffle(test_indices_corpus3)
        np.random.shuffle(train_indices_corpus3)

    test_corpus = [corpus1[i] for i in test_indices_corpus1] + [corpus2[i] for i in test_indices_corpus2]
    train_corpus = [corpus1[i] for i in train_indices_corpus1] + [corpus2[i] for i in train_indices_corpus2]
    if args.corpus3_path:
        test_corpus += [corpus3[i] for i in test_indices_corpus3]
        train_corpus += [corpus3[i] for i in train_indices_corpus3]

    with open(os.path.join(args.output_path, 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_corpus)
    with open(os.path.join(args.output_path, 'test.txt'), 'w', encoding='utf-8') as f:
        f.writelines(test_corpus)

    print('train.txt and test.txt written to {}'.format(args.output_path))
