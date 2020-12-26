import os
import pickle
import warnings
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from docopt import docopt
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import log_softmax
from transformers import BertTokenizer, BertForMaskedLM
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
    window_size = int((sequence_length - 2) / 2)

    # determine where context starts and if there are any unused context positions to the left
    if target_position - window_size >= 0:
        start = target_position - window_size
        extra_left = 0
    else:
        start = 0
        extra_left = window_size - target_position

    # determine where context ends and if there are any unused context positions to the right
    if target_position + window_size + 1 <= len(token_ids):
        end = target_position + window_size + 1
        extra_right = 0
    else:
        end = len(token_ids)
        extra_right = target_position + window_size + 1 - len(token_ids)

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

    new_target_position = target_position - start + 1

    return item, new_target_position


class ContextsDataset(torch.utils.data.Dataset):

    def __init__(self, targets_i2w, sentences, context_size, tokenizer, n_sentences=None):
        super(ContextsDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.context_size = context_size

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sentence in tqdm(sentences, total=n_sentences):
                token_ids = tokenizer.encode(' '.join(sentence), add_special_tokens=False)
                for spos, tok_id in enumerate(token_ids):
                    if tok_id in targets_i2w:
                        model_input, pos_in_context = get_context(tokenizer, token_ids, spos, context_size)
                        self.data.append((model_input, targets_i2w[tok_id], pos_in_context))

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


def to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.detach().cpu().clone().numpy()
    else:
        return tensor.clone().numpy()


def main():
    """
    Collect lexical substitutes and their probabilities.
    """

    # Get the arguments
    args = docopt("""Collect BERT representations from corpus.

    Usage:
        substitutes.py [--nSubs=50 --context=64 --batch=64 --localRank=-1 --ignoreBias] <modelName> <corpDir> <testSet> <outPath>

    Arguments:
        <modelName> = HuggingFace model name 
        <corpDir> = path to corpus or corpus directory (iterates through files)
        <testSet> = path to file with one target per line
        <outPath> = output path for substitutes

    Options:
        --nSubs=N  The number of lexical substitutes to extract [default: 100]
        --context=C  The length of a token's entire context window [default: 64]
        --batch=B  The batch size [default: 64]
        --localRank=R  For distributed training [default: -1]
        --ignoreBias  Whether to ignore the bias vector during masked word prediction [default=False]
    """)

    modelName = args['<modelName>']
    corpDir = args['<corpDir>']
    testSet = args['<testSet>']
    outPath = args['<outPath>']
    nSubs = int(args['--nSubs'])
    contextSize = int(args['--context'])
    batchSize = int(args['--batch'])
    localRank = int(args['--localRank'])
    ignore_lm_bias = bool(args['--ignoreBias'])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    # Setup CUDA, GPU & distributed training
    if localRank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(localRank)
        device = torch.device("cuda", localRank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if localRank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        localRank,
        device,
        n_gpu,
        bool(localRank != -1)
    )

    # Set seeds across modules
    set_seed(42, n_gpu)

    # Load targets
    targets = []
    with open(testSet, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            target = line.strip()
            targets.append(target)
    print('=' * 80)
    print('targets:', targets)
    print('=' * 80)

    # Load pretrained model and tokenizer
    if localRank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(modelName, never_split=targets)
    model = BertForMaskedLM.from_pretrained(modelName, output_hidden_states=True)

    if ignore_lm_bias:
        logger.warning('Ignoring bias vector for masked word prediction.')
        model.cls.predictions.decoder.bias = None

    if localRank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    # Store vocabulary indices of target words
    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    assert len(targets) == len(targets_ids)
    i2w = {}
    for t, t_id in zip(targets, targets_ids):
        if len(t_id) > 1 or (len(t_id) == 1 and t_id == tokenizer.unk_token_id):
            if tokenizer.add_tokens([t]):
                model.resize_token_embeddings(len(tokenizer))
                i2w[len(tokenizer) - 1] = t
            else:
                logger.error('Word not properly added to tokenizer:', t, tokenizer.tokenize(t))
        else:
            i2w[t_id[0]] = t

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if localRank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[localRank], output_device=localRank, find_unused_parameters=True
        )

    # Get sentence iterator
    sentences = PathLineSentences(corpDir)

    # with warnings.catch_warnings():
    #     warnings.resetwarnings()
    #     warnings.simplefilter("always")
    nSentences = 0
    target_counter = {target: 0 for target in i2w}
    for sentence in sentences:
        nSentences += 1
        for tok_id in tokenizer.encode(' '.join(sentence), add_special_tokens=False):
            if tok_id in target_counter:
                target_counter[tok_id] += 1

    logger.warning('usages: %d' % (sum(list(target_counter.values()))))

    def collate(batch):
        return [
            {'input_ids': torch.cat([item[0]['input_ids'] for item in batch], dim=0),
             'attention_mask': torch.cat([item[0]['attention_mask'] for item in batch], dim=0)},
            [item[1] for item in batch],
            [item[2] for item in batch]
        ]

    # Container for lexical substitutes
    substitutes = {
        i2w[target]: [{} for _ in range(target_count)]
        for (target, target_count) in target_counter.items()
    }

    # Iterate over sentences and collect representations
    nUsages = 0
    curr_idx = {i2w[target]: 0 for target in target_counter}

    dataset = ContextsDataset(i2w, sentences, contextSize, tokenizer, nSentences)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batchSize, collate_fn=collate)
    iterator = tqdm(dataloader, desc="Iteration", disable=localRank not in [-1, 0])

    for step, batch in enumerate(iterator):
        model.eval()

        inputs, lemmas, positions = batch
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        bsz = inputs['input_ids'].shape[0]

        with torch.no_grad():
            outputs = model(**inputs)  # n_sentences, max_sent_len, vocab_size

            logits = outputs[0][np.arange(bsz), positions, :]  # n_sentences, vocab_size
            logp = log_softmax(logits, dim=-1)
            values, indices = torch.sort(logp, dim=-1, descending=True)
            values, indices = values[:, :nSubs], indices[:, :nSubs]  # n_sentences, n_substitutes

            hidden_states = outputs[1]

            input_ids = to_numpy(inputs['input_ids'])
            attention_mask = to_numpy(inputs['attention_mask'])
            values = to_numpy(values)
            indices = to_numpy(indices)
            last_layer = to_numpy(hidden_states[-1][np.arange(bsz), positions, :])

            for b_id in np.arange(bsz):
                lemma = lemmas[b_id]

                substitutes[lemma][curr_idx[lemma]]['candidates'] = indices[b_id]
                substitutes[lemma][curr_idx[lemma]]['logp'] = values[b_id]
                substitutes[lemma][curr_idx[lemma]]['input_ids'] = input_ids[b_id]
                substitutes[lemma][curr_idx[lemma]]['attention_mask'] = attention_mask[b_id]
                substitutes[lemma][curr_idx[lemma]]['position'] = positions[b_id]
                substitutes[lemma][curr_idx[lemma]]['embedding'] = last_layer[b_id, :] # / last_layer[b_id, :].sum()

                curr_idx[lemma] += 1
                nUsages += 1

    iterator.close()
    with open(outPath, 'wb') as f_out:
        pickle.dump(substitutes, f_out)

    logger.warning('usages: %d' % (nUsages))
    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
