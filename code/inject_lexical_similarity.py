import argparse
import gensim.downloader
import json
import pickle
import torch
import time
import logging
import numpy as np
from smart_open import open
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


BERT_UNK = '[UNK]'
ELMO_UNK = '<UNK>'


class SubstitutesDataset(torch.utils.data.Dataset):

    def __init__(self, substitutes_raw, tokenizer, normalise_embeddings):
        super(SubstitutesDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer

        for target in substitutes_raw:
            target_id = self.tokenizer.convert_tokens_to_ids(target)
            target_occurrence_idx = 0
            for occurrence in substitutes_raw[target]:
                candidate_tokens = tokenizer.convert_ids_to_tokens(occurrence['candidates'])

                if normalise_embeddings:
                    embedding = normalize(torch.tensor(occurrence['embedding']).unsqueeze(0), p=2)[0]
                else:
                    embedding = occurrence['embedding']

                for j in range(len(occurrence['candidates'])):
                    candidate_id = torch.as_tensor(occurrence['candidates'][j])
                    candidate_token = candidate_tokens[j]

                    # target is most often among candidates - skip it
                    if candidate_id == target_id:
                        continue

                    # skip punctuation and similar
                    if not any(c.isalpha() for c in candidate_token):
                        continue

                    input_ids = occurrence['input_ids'].copy()
                    input_ids[occurrence['position']] = candidate_id
                    inputs = {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                              'attention_mask': torch.tensor(occurrence['attention_mask'], dtype=torch.long)}

                    self.data.append((
                        inputs,
                        target,
                        target_occurrence_idx,
                        candidate_token,
                        embedding,
                        occurrence['logp'][j],
                        occurrence['position']
                    ))

                target_occurrence_idx += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Modify substitute probabilities based on lexical similarity with target.
    """
    parser = argparse.ArgumentParser(
        description='Modify substitute probabilities based on lexical similarity with target.')
    parser.add_argument(
        '--model_name', type=str, required=True,
        help='HuggingFace model name or path')
    parser.add_argument(
        '--static_model_name', type=str, required=True,
        help='Gensim model name or path')
    parser.add_argument(
        '--subs_path', type=str, required=True,
        help='Path to the pickle file containing substitute lists (output by substitutes.py).')
    parser.add_argument(
        '--targets_path', type=str, required=True,
        help='Path to the csv file containing target word forms.')
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output path for pickle containing substitutes with lexical similarity values.')
    parser.add_argument(
        '--model_type', type=str, required=True, choices=['elmo', 'bert'],
        help='LSTM or Transformer language model.')
    parser.add_argument(
        '--ignore_unk', action='store_true',
        help='Whether to remove the UNK token from candidate lists.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    # Load target forms
    target_forms = []
    with open(args.targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            forms = line.split(',')[1:]
            target_forms.extend(forms)
    print('=' * 80)
    print('targets:', target_forms)
    print('=' * 80)

    # # Load model and tokenizer
    # if args.candidates_as_bert_ids:
    #     logger.warning('Loading BERT tokenizer.')
    #     tokenizer = BertTokenizer.from_pretrained(args.model_name, never_split=target_forms, use_fast=False)

    logger.warning('Loading Gensim model.')
    try:
        static_model = gensim.downloader.load(args.static_model_name)
    except ValueError:
        static_model = gensim.models.KeyedVectors.load(args.static_model_name)

    # # Store vocabulary indices of target words
    # targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in target_forms]
    # assert len(target_forms) == len(targets_ids)
    # words_added = []
    # for t, t_id in zip(target_forms, targets_ids):
    #     if tokenizer.do_lower_case:
    #         t = t.lower()
    #     if t in tokenizer.added_tokens_encoder:
    #         continue
    #     if len(t_id) > 1 or (len(t_id) == 1 and t_id[0] == tokenizer.unk_token_id):
    #         if tokenizer.add_tokens([t]):
    #             words_added.append(t)
    #         else:
    #             logger.error('Word not properly added to tokenizer:', t, tokenizer.tokenize(t))
    #
    # # check if correctly added
    # for t, t_id in zip(target_forms, targets_ids):
    #     if len(t_id) != 1:
    #         print(t, t_id)
    # logger.warning("\nTarget words added to the vocabulary: {}.\n".format(', '.join(words_added)))



    if args.subs_path.endswith('.pkl'):
        with open(args.subs_path, 'rb') as f_in:
            substitutes_raw = pickle.load(f_in)
    elif args.subs_path.endswith('.json'):
        with open(args.subs_path, 'r') as f_in:
            substitutes_raw = json.load(f_in)
    else:
        raise ValueError('Invalid path: {}'.format(args.subs_path))

    substitutes_new = {
        w: [{'candidate_words': [], 'logp': [], 'dot_products': []} for _ in substitutes_raw[w]]
        for w in substitutes_raw
    }

    for target in substitutes_raw:

        occurrence_idx = 0
        for occurrence in substitutes_raw[target]:
            if args.model_type == 'bert':
                # compute and store in substitutes_new
                compute_lexical_similarity_bert(occurrence, occurrence_idx, target, static_model, substitutes_new, args)
            else:
                # compute and store in substitutes_new, merging backward and forward
                compute_lexical_similarity_elmo(occurrence, occurrence_idx, target, static_model, substitutes_new, args)
            occurrence_idx += 1

    if args.output_path.endswith('.pkl'):
        with open(args.output_path, 'wb') as f_out:
            pickle.dump(substitutes_new, f_out)
    elif args.output_path.endswith('.json'):
        raise NotImplementedError()
        # with open(args.output_path, 'w', encoding='utf-8') as f_out:
        #     out = json.dumps(substitutes_new, ensure_ascii=False, sort_keys=True, indent=4)
        #     f_out.write(out)
    else:
        raise ValueError('Invalid output path: {}'.format(args.output_path))


    logger.warning("--- %s seconds ---" % (time.time() - start_time))


def compute_lexical_similarity_elmo(occurrence, occurrence_idx, target, static_model, substitutes_new, args):

    seen = {}  # maps candidates seen in backward to their index in the merged candidate list

    for dir in ['backward', 'forward']:
        candidate_tokens = occurrence[dir]['candidate_words']

        for j in range(len(occurrence[dir]['candidate_words'])):

            # target is most often among candidates - skip it
            if candidate_tokens[j] == target:
                continue

            # skip punctuation and similar
            if not any(c.isalpha() for c in candidate_tokens[j]):
                continue

            if args.ignore_unk and candidate_tokens[j] == ELMO_UNK:
                continue

            if candidate_tokens[j] not in seen:
                try:
                    dot_product = static_model.similarity(target, candidate_tokens[j])
                    assert (dot_product <= 1.01)
                except KeyError:
                    # e.g. word '##ing' not in vocabulary
                    dot_product = 0.

                seen[candidate_tokens[j]] = len(substitutes_new[target][occurrence_idx]['candidate_words'])

                substitutes_new[target][occurrence_idx]['candidate_words'].append(candidate_tokens[j])
                substitutes_new[target][occurrence_idx]['logp'].append(occurrence[dir]['logp'][j])
                substitutes_new[target][occurrence_idx]['dot_products'].append(dot_product)

            else:
                prev_idx = seen[candidate_tokens[j]]
                logp_backward = substitutes_new[target][occurrence_idx]['logp'][prev_idx]
                logp_forward = occurrence[dir]['logp'][j]
                logp_merged = np.log(np.exp(logp_backward) + np.exp(logp_forward))
                substitutes_new[target][occurrence_idx]['logp'][prev_idx] = logp_merged

    # re-normalise
    log_denominator = np.log(np.sum(np.exp(substitutes_new[target][occurrence_idx]['logp'])))  # .ln()
    for logp in substitutes_new[target][occurrence_idx]['logp']:
        logp -= log_denominator


def compute_lexical_similarity_bert(occurrence, occurrence_idx, target, static_model, substitutes_new, args):

    candidate_tokens = occurrence['candidate_words']

    for j in range(len(occurrence['candidate_words'])):

        # target is most often among candidates - skip it
        if candidate_tokens[j] == target:
            continue
        # skip punctuation and similar
        if not any(c.isalpha() for c in candidate_tokens[j]):
            continue

        if args.ignore_unk and candidate_tokens[j] == BERT_UNK:
            continue

        try:
            dot_product = static_model.similarity(target, candidate_tokens[j])
            assert (-1 <= dot_product <= 1, dot_product)
        except KeyError:
            # e.g. word '##ing' not in vocabulary
            dot_product = 0.

        substitutes_new[target][occurrence_idx]['candidate_words'].append(candidate_tokens[j])
        substitutes_new[target][occurrence_idx]['logp'].append(occurrence['logp'][j])
        substitutes_new[target][occurrence_idx]['dot_products'].append(dot_product)


if __name__ == '__main__':
    main()
