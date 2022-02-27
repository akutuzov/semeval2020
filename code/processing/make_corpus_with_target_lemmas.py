import argparse
from collections import defaultdict
from gensim.models.word2vec import PathLineSentences
from docopt import docopt
import logging
import time
from smart_open import open


def convert(data_path=None, output_path=None, targets_path=None):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # Get sentence iterator
    sentences = PathLineSentences(data_path)

    # Load targets
    form2target = {}
    target_forms = []
    with open(targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            entries = line.split(',')
            target, forms = entries[0], entries[1:]
            target_forms.extend(forms)
            for form in forms:
                form2target[form] = target

    print('=' * 80)
    print('targets:', target_forms)
    print(form2target)
    print('=' * 80)

    # Iterate over sentences and words
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sentence in sentences:
            new_sentence = []
            for word in sentence:
                if word in form2target:
                    new_sentence.append(form2target[word])
                else:
                    new_sentence.append(word)
            f_out.write(' '.join(new_sentence) + '\n')


if __name__ == '__main__':
    # for lang in ['english', 'latin', 'german', 'swedish']:
    #     for corpus in ['corpus1', 'corpus2']:
    #         main(data_path='/Volumes/Disk1/SemEval/finetuning_corpora/{}/token/{}.txt.gz'.format(lang, corpus),
    #              output_path='/Volumes/Disk1/SemEval/finetuning_corpora/{}/token/{}.lemma.txt.gz'.format(lang, corpus),
    #              targets_path='/Volumes/Disk1/SemEval/finetuning_corpora/{}/targets/target_forms.csv'.format(lang))
    #
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
     '--targets_path', type=str, required=True,
     help='Path to the csv file containing target word forms.'
    )
    args = parser.parse_args()

    convert(data_path = args.data_path, output_path = args.output_path, targets_path =
            args.targets_path )

