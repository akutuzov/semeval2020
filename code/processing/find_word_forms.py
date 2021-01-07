import argparse
import logging
import stanza
import time
from gensim import utils as gensim_utils
from tqdm import tqdm

logger = logging.getLogger(__name__)

def pos_match(target_pos, stanza_pos):
    if target_pos == 'nn' and stanza_pos == 'NOUN':
        return True
    elif target_pos == 'vb' and stanza_pos == 'VERB':
        return True
    else:
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", type=str, required=True,
        help="Path of the .bz2, .gz, or text file containing the training split of the corpus, or the whole corpus."
    )
    parser.add_argument(
        "--test_path", type=str,
        help="Optional path of the .bz2, .gz, or text file containing the test split of the corpus."
    )
    parser.add_argument(
        "--targets_path", type=str, required=True,
        help="Path of the text file containing one target word per line."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path for the output csv file."
    )
    parser.add_argument(
        "--lang", type=str, required=True,
        help="The language code for the corpus, used to select the correct lemmatiser."
    )
    args = parser.parse_args()

    lang = args.lang.lower()
    assert lang in ['en', 'de', 'sv', 'la', 'ru', 'it']

    if lang == 'en':
        processors = 'tokenize, lemma, pos'
    else:
        processors = 'tokenize, lemma'

    try:
        nlp = stanza.Pipeline(lang=lang, processors=processors)
    except Exception:
        stanza.download(lang=lang, processors=processors)
        nlp = stanza.Pipeline(lang=lang, processors=processors)

    # Load targets
    targets = []
    if lang == 'en':
        pos_tags = {}
    with open(args.targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            target = line.strip()
            if not target:
                continue

            if lang == 'en':
                lemma_pos = target.split('_')
                lemma, pos = lemma_pos[0], lemma_pos[1]
                targets.append(lemma)
                pos_tags[lemma] = pos
            else:
                targets.append(target)
    logger.warning('\nTarget words:')
    logger.warning('{}.\n'.format(', '.join(targets)))
    
    target_lemmas = {nlp(w).sentences[0].words[0].lemma: w for w in targets}
    logger.warning('\nTarget lemmas:')
    logger.warning('{}.\n'.format(', '.join(targets)))

    if lang == 'en':
        pos_tags = {w: pos_tags[target_lemmas[w]] for w in target_lemmas}

    start_time = time.time()

    logger.warning('Lemmatize corpus...')
    all_forms = {lemma: {lemma} for lemma in target_lemmas}
    for path in [args.train_path, args.test_path]:
        if not path:
            continue

        with gensim_utils.file_or_filename(path) as f:
            n_lines = 0
            for line in f:
                n_lines += 1

        with gensim_utils.file_or_filename(path) as f:
            lines = ''
            for line in tqdm(f, total=n_lines):
                line = gensim_utils.to_unicode(line, encoding='utf-8')
                for sentence in nlp(line).sentences:
                    for w in sentence.words:
                        if w.lemma in target_lemmas:
                            if lang != 'en' or (lang == 'en' and pos_match(pos_tags[w.lemma], w.pos)):
                                all_forms[w.lemma].add(w.text)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))

    with open(args.output_path, 'w') as f:
        for lemma in target_lemmas:
            orig_target = target_lemmas[lemma]
            forms = list(all_forms[lemma])
            f.write('{}\n'.format(', '.join([orig_target] + forms)))

    logger.warning('Output written to {}'.format(args.output_path))
