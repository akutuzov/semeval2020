import os
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from docopt import docopt
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer, BertModel
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
                    line = gensim_utils.to_unicode(line, encoding='latin-1').split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


def get_context(token_ids, target_position, sequence_length):
    """
    Given a text containing a target word, return the sentence snippet which surrounds the target word
    (and the target word's position in the snippet).

    :param token_ids: list of token ids (for an entire line of text)
    :param target_position: index of the target word's position in `tokens`
    :param sequence_length: desired length for output sequence (e.g. 128, 256, 512)
    :return: (context_ids, new_target_position)
                context_ids: list of token ids for the output sequence
                new_target_position: index of the target word's position in `context_ids`
    """
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position


class ContextsDataset(torch.utils.data.Dataset):

    def __init__(self, targets_i2w, sentences, context_size, tokenizer, n_sentences=None):
        super(ContextsDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.CLS_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0]
        self.SEP_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]

        for s_id, sentence in enumerate(tqdm(sentences, total=n_sentences)):
            token_ids = tokenizer.encode(' '.join(sentence), add_special_tokens=False)
            for spos, tok_id in enumerate(token_ids):
                if tok_id in targets_i2w:
                    context_ids, pos_in_context = get_context(token_ids, spos, context_size)
                    input_ids = [self.CLS_id] + context_ids + [self.SEP_id]
                    self.data.append((input_ids, targets_i2w[tok_id], pos_in_context))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, lemma, pos_in_context = self.data[index]
        return torch.tensor(input_ids), lemma, pos_in_context


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Collect BERT representations from corpus.
    """

    # Get the arguments
    args = docopt("""Collect BERT representations from corpus.

    Usage:
        collect.py [--context=64 --batch=64 --localRank=-1] <modelConfig> <corpDir> <testSet> <outPath>

    Arguments:
        <modelConfig> = path to file with model name, number of layers, and layer dimensionality (space-separated)    
        <corpDir> = path to corpus or corpus directory (iterates through files)
        <testSet> = path to file with one target per line
        <outPath> = output path for usage matrices

    Options:
        --context=N  The length of a token's entire context window [default: 64]
        --batch=B  The batch size [default: 64]
        --localRank=R  For distributed training [default: -1]
    """)

    corpDir = args['<corpDir>']
    testSet = args['<testSet>']
    outPath = args['<outPath>']
    contextSize = int(args['--context'])
    batchSize = int(args['--batch'])
    localRank = int(args['--localRank'])
    with open(args['<modelConfig>'], 'r', encoding='utf-8') as f_in:
        modelConfig = f_in.readline().split()
        modelName, nLayers, nDims = modelConfig[0], int(modelConfig[1]), int(modelConfig[2])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
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

    # Set seed
    set_seed(42, n_gpu)

    # Load pretrained model and tokenizer
    if localRank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(modelName)
    model = BertModel.from_pretrained(modelName, output_hidden_states=True)

    if localRank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    # Load targets
    targets = []
    with open(testSet, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            target = line.strip()
            try:
                lemma_pos = target.split('_')
                lemma, pos = lemma_pos[0], lemma_pos[1]
                targets.append(lemma)
            except IndexError:
                targets.append(target)

    print('='*80)
    print('targets:', targets)
    print('=' * 80)

    # Store vocabulary indices of target words
    i2w = {}
    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    assert len(targets) == len(targets_ids)
    for t, t_id in zip(targets, targets_ids):
        if len(t_id) > 1:
            tokenizer.add_tokens([t])
            model.resize_token_embeddings(len(tokenizer))
            i2w[len(tokenizer) - 1] = t
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

    nSentences = 0
    target_counter = {target: 0 for target in i2w}
    for sentence in sentences:
        nSentences += 1
        for tok_id in tokenizer.encode(' '.join(sentence), add_special_tokens=False):
            if tok_id in target_counter:
                target_counter[tok_id] += 1

    # Container for usages
    usages = {
        i2w[target]: np.empty((target_count, nLayers * nDims))  # usage matrix
        for (target, target_count) in target_counter.items()
    }

    # Iterate over sentences and collect representations
    nUsages = 0
    curr_idx = {i2w[target]: 0 for target in target_counter}

    dataset = ContextsDataset(i2w, sentences, contextSize, tokenizer, nSentences)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batchSize)
    iterator = tqdm(dataloader, desc="Iteration", disable=localRank not in [-1, 0])

    for step, batch in enumerate(iterator):
        model.eval()
        batch_tuple = tuple()
        for t in batch:
            try:
                batch_tuple += (t.to(device),)
            except AttributeError:
                batch_tuple += (t,)

        batch_input_ids = batch_tuple[0].squeeze(1)
        batch_lemmas, batch_spos = batch_tuple[1], batch_tuple[2]

        with torch.no_grad():
            if torch.cuda.is_available():
                batch_input_ids = batch_input_ids.to('cuda')

            outputs = model(batch_input_ids)

            if torch.cuda.is_available():
                hidden_states = [l.detach().cpu().clone().numpy() for l in outputs[2]]
            else:
                hidden_states = [l.clone().numpy() for l in outputs[2]]

            # store usage tuples in a dictionary: lemma -> (vector, position)
            for b_id in np.arange(len(batch_input_ids)):
                lemma = batch_lemmas[b_id]

                layers = [layer[b_id, batch_spos[b_id] + 1, :] for layer in hidden_states]
                usage_vector = np.concatenate(layers)
                usages[lemma][curr_idx[lemma], :] = usage_vector

                curr_idx[lemma] += 1
                nUsages += 1

    iterator.close()
    np.savez_compressed(outPath, **usages)

    logging.info('usages: %d' % (nUsages))
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
