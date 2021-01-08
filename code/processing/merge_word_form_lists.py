import argparse
import os
from collections import defaultdict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_path", type=str, required=True,
        help="Path of the directory containing the lists of target word forms to merge. Also used as output path."
    )
    args = parser.parse_args()

    files = [f for f in os.listdir(args.dir_path) if f.startswith('target_forms_') and f.endswith('.csv')]

    word_forms = defaultdict(set)
    for file in files:
        with open(os.path.join(args.dir_path, file), 'r') as f:
            for line in f:
                line = line.strip()
                entries = line.split(', ')
                target, forms = entries[0], entries[1:]

                for form in forms:
                    word_forms[target].add(form)

    with open(os.path.join(args.dir_path, 'target_forms.csv'), 'w') as f:
        for target in word_forms:
            f.write('{}\n'.format(','.join([target] + list(word_forms[target]))))
