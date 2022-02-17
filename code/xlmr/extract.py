import argparse
import os
import warnings
from collections import defaultdict
from smart_open import open
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForMaskedLM, AutoTokenizer
from gensim import utils as gensim_utils

logger = logging.getLogger(__name__)


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
            with gensim_utils.file_or_filename(file_name) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = gensim_utils.to_unicode(line, encoding='utf-8').split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


def get_context(tokenizer, token_ids, target_position, sequence_length):
    """
        Given a text containing a target word, return the sentence snippet which surrounds the target word
        (and the target word's position in the snippet).

        :param tokenizer: a Huggingface tokenizer
        :param token_ids: list of token ids for the entire context sentence
        :param target_position: tuple with the target word's start and end position in `token_ids`,
                                such that token_ids[target_position[0]:target_position[1]] = target_word_ids
        :param sequence_length: desired length for output sequence (e.g., 128, 256, 512)
        :return: (context_ids, new_target_position)
                    context_ids: list of token ids in the target word's context window
                    new_target_position: tuple with the target word's start and end position in `context_ids`
    """

    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 4) / 2)
    target_len = target_position[1] - target_position[0]

    if target_len % 2 == 0:
        window_size_left = int(window_size - target_len / 2) + 1
        window_size_right = int(window_size - target_len / 2)
    else:
        window_size_left = int(window_size - target_len // 2)
        window_size_right = int(window_size - target_len // 2)

    # determine where context starts and if there are any unused context positions to the left
    if target_position[0] - window_size_left >= 0:
        start = target_position[0] - window_size_left
        extra_left = 0
    else:
        start = 0
        extra_left = window_size_left - target_position[0]

    # determine where context ends and if there are any unused context positions to the right
    if target_position[1] + window_size_right + 1 <= len(token_ids):
        end = target_position[1] + window_size_right + 1
        extra_right = 0
    else:
        end = len(token_ids)
        extra_right = target_position[1] + window_size_right + 1 - len(token_ids)

    # redistribute to the left the unused right context positions
    if extra_right > 0 and extra_left == 0:
        if start - extra_right >= 0:
            padding = 0
            start -= extra_right
        else:
            padding = extra_right - start
            start = 0
    # redistribute to the right the unused left context positions
    elif extra_left > 0 and extra_right == 0:
        if end + extra_left <= len(token_ids):
            padding = 0
            end += extra_left
        else:
            padding = end + extra_left - len(token_ids)
            end = len(token_ids)
    else:
        padding = extra_left + extra_right

    context_ids = token_ids[start:end]
    context_ids = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id]
    item = {'input_ids': context_ids + padding * [tokenizer.pad_token_id],
            'attention_mask': len(context_ids) * [1] + padding * [0]}

    new_target_position = (target_position[0] - start + 1, target_position[1] - start + 1)

    return item, new_target_position


class ContextsDataset(torch.utils.data.Dataset):

    def __init__(self, targets_i2w, sentences, context_size, tokenizer, len_longest_tokenized=10, n_sentences=None):
        super(ContextsDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.context_size = context_size

        with warnings.catch_warnings():
            for sentence in tqdm(sentences, total=n_sentences):
                sentence_token_ids_full = tokenizer.encode(' '.join(sentence), add_special_tokens=False)
                sentence_token_ids = list(sentence_token_ids_full)
                while sentence_token_ids:
                    candidate_ids_found = False
                    for length in list(range(1, len_longest_tokenized + 1))[::-1]:
                        candidate_ids = tuple(sentence_token_ids[-length:])
                        if candidate_ids in targets_i2w:
                            sent_position = (len(sentence_token_ids) - length, len(sentence_token_ids))

                            context_ids, pos_in_context = get_context(
                                tokenizer, sentence_token_ids_full, sent_position, context_size)
                            self.data.append((context_ids, targets_i2w[candidate_ids], pos_in_context))

                            sentence_token_ids = sentence_token_ids[:-length]
                            candidate_ids_found = True
                            break
                    if not candidate_ids_found:
                        sentence_token_ids = sentence_token_ids[:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        model_input, lemma, pos_in_context = self.data[index]
        model_input = {'input_ids': torch.tensor(model_input['input_ids'], dtype=torch.long).unsqueeze(0),
                       'attention_mask': torch.tensor(model_input['attention_mask'], dtype=torch.long).unsqueeze(0)}
        return model_input, lemma, pos_in_context


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Collect XLM-R representations from corpus.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path', type=str, required=True, 
        help='path to model directory or model name (e.g., xlm-roberta-base)'
    )
    parser.add_argument(
        '--targets_path', type=str, required=True,
        help='Path to file with target words (one word per line — possibly with tab-separated change score — '
             'or a list of comma-separated word forms.'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output path for extracted embeddings.'
    )
    parser.add_argument(
        '--corpus_path', type=str, required=True,
        help='Path to corpus or corpus directory (iterates through files).'
    )
    parser.add_argument(
        '--context_window', type=int, default=512,
        help="The length of a token's entire context window"
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='The number of sentences processed at once by the LM.'
    )
    # parser.add_argument(
    #     '--n_layers', type=int, default=12,
    #     help='The number of layers of the Transformer model.'
    # )
    parser.add_argument(
        '--n_dims', type=int, default=768,
        help='The dimensionality of a Transformer layer (hence the dimensionality of the output embeddings).'
    )
    parser.add_argument(
        '--local_rank', type=int, default=-1,
        help='For distributed training (default: -1).'
    )

    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        n_gpu,
        bool(args.local_rank != -1)
    )

    # Set seeds across modules
    set_seed(42, n_gpu)

    # Load targets
    targets = defaultdict(list)
    with open(args.targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            forms = line.split(',')
            if len(forms) > 1:
                for form in forms:
                    if form not in targets[forms[0]]:
                        targets[forms[0]].append(form)
            else:
                line = line.split('\t')
                targets[line[0]].append(line[0])
                
    n_target_forms = sum([len(vals) for vals in targets.values()])
    logger.warning(f"Target lemmas: {len(targets)}.")
    logger.warning(f"Target word forms: {n_target_forms}.")

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) #, never_split=targets)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, output_hidden_states=True)

    logger.warning(f"Tokenizer's added tokens:\n{tokenizer.get_added_vocab()}")

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    # Store vocabulary indices of target words
    targets_ids = defaultdict(lambda: dict())
    for lemma in targets:
        for form in targets[lemma]:
            targets_ids[lemma][form] = tokenizer.encode(form, add_special_tokens=False)
            
    assert n_target_forms == sum([len(vals) for vals in targets_ids.values()])

    ids2lemma = {}  # maps all forms' token ids to their corresponding lemma
    lemma2ids = defaultdict(list)  # maps every lemma to a list of token ids corresponding to all word forms
    len_longest_tokenized = 0

    for lemma, forms2ids in targets_ids.items():
        for form, form_id in forms2ids.items():

            # remove '▁' from the beginning of subtoken sequences
            if len(form_id) > 1 and form_id[0] == 6:
                form_id = form_id[1:]

            if len(form_id) == 0:
                logger.warning('Empty string? Lemma: {}\tForm:"{}"\tTokenized: "{}"'.format(
                    lemma, form, tokenizer.tokenize(form)))
                continue

            if len(form_id) == 1 and form_id[0] == tokenizer.unk_token_id:
                logger.warning('Tokenizer returns UNK for this word form. '
                               'Lemma: {}\tForm: {}\tTokenized: {}'.format(lemma, form, tokenizer.tokenize(form)))
                continue

            if len(form_id) > 1:
                logger.warning('Word form split into subtokens. '
                               'Lemma: {}\tForm: {}\tTokenized: {}'.format(lemma, form, tokenizer.tokenize(form)))

            ids2lemma[tuple(form_id)] = lemma
            lemma2ids[lemma].append(tuple(form_id))
            if len(tuple(form_id)) > len_longest_tokenized:
                len_longest_tokenized = len(tuple(form_id))

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Get sentence iterator
    sentences = PathLineSentences(args.corpus_path)

    nSentences = 0
    target_counter = {target: 0 for target in lemma2ids}
    for sentence in sentences:
        nSentences += 1
        sentence_token_ids = tokenizer.encode(' '.join(sentence), add_special_tokens=False)

        while sentence_token_ids:
            candidate_ids_found = False
            for length in list(range(1, len_longest_tokenized + 1))[::-1]:
                candidate_ids = tuple(sentence_token_ids[-length:])
                if candidate_ids in ids2lemma:
                    target_counter[ids2lemma[candidate_ids]] += 1
                    sentence_token_ids = sentence_token_ids[:-length]
                    candidate_ids_found = True
                    break
            if not candidate_ids_found:
                sentence_token_ids = sentence_token_ids[:-1]

    logger.warning('Total usages: %d' % (sum(list(target_counter.values()))))

    for lemma in target_counter:
        logger.warning(f'{lemma}: {target_counter[lemma]}')

    # Container for usages
    usages = {
        target: np.empty((target_count, args.n_dims))  # usage matrix
        for (target, target_count) in target_counter.items()
    }

    # Iterate over sentences and collect representations
    nUsages = 0
    curr_idx = {target: 0 for target in target_counter}

    def collate(batch):
        return [
            {'input_ids': torch.cat([item[0]['input_ids'] for item in batch], dim=0),
             'attention_mask': torch.cat([item[0]['attention_mask'] for item in batch], dim=0)},
            [item[1] for item in batch],
            [item[2] for item in batch]
        ]

    dataset = ContextsDataset(ids2lemma, sentences, args.context_window, tokenizer, len_longest_tokenized, nSentences)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
    iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(iterator):
        model.eval()
        batch_tuple = tuple()
        for t in batch:
            try:
                batch_tuple += (t.to(device),)
            except AttributeError:
                batch_tuple += (t,)

        batch_input_ids = batch_tuple[0]
        batch_lemmas, batch_spos = batch_tuple[1], batch_tuple[2]

        with torch.no_grad():
            if torch.cuda.is_available():
                batch_input_ids = batch_input_ids.to('cuda')

            outputs = model(**batch_input_ids)

            if torch.cuda.is_available():
                hidden_states = [l.detach().cpu().clone().numpy() for l in outputs.hidden_states]
            else:
                hidden_states = [l.clone().numpy() for l in outputs.hidden_states]

            # store usage tuples in a dictionary: lemma -> (vector, position)
            for b_id in np.arange(len(batch_lemmas)):
                lemma = batch_lemmas[b_id]
                layers = [layer[b_id, batch_spos[b_id][0]:batch_spos[b_id][1], :] for layer in hidden_states]
                usage_vector = np.mean(layers, axis=0)
                if usage_vector.shape[0] > 1:
                    usage_vector = np.mean(usage_vector, axis=0)
                usages[lemma][curr_idx[lemma], :] = usage_vector

                curr_idx[lemma] += 1
                nUsages += 1

    iterator.close()
    np.savez_compressed(args.output_path, **usages)

    logger.warning('Total embeddings: %d' % (nUsages))
    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
