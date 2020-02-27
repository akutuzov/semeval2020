import numpy as np
import torch
from gensim.models.word2vec import PathLineSentences
from docopt import docopt
import logging
import time
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


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


def main():
    """
    Collect BERT representations from corpus.
    """

    # Get the arguments
    args = docopt("""Collect BERT representations from corpus.

    Usage:
        collect.py [--context=64 --buffer=64] <modelConfig> <corpDir> <testSet> <outPath>

    Arguments:
        <modelConfig> = path to file with model name, number of layers, and layer dimensionality (space-separated)    
        <corpDir> = path to corpus or corpus directory (iterates through files)
        <testSet> = path to file with one target per line
        <outPath> = output path for usage matrices

    Options:
        --context=N  The length of a token's entire context window [default: 64]
        --buffer=B  The number of usages to process with a single model execution [default: 64]

    """)

    corpDir = args['<corpDir>']
    testSet = args['<testSet>']
    outPath = args['<outPath>']
    contextSize = int(args['--context'])
    bufferSize = int(args['--buffer'])
    with open(args['<modelConfig>'], 'r', encoding='utf-8') as f_in:
        modelConfig = f_in.readline().split()
        modelName, nLayers, nDims = modelConfig[0], int(modelConfig[1]), int(modelConfig[2])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
    logging.info(__file__.upper())
    start_time = time.time()


    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(modelName)
    model = BertModel.from_pretrained(modelName, output_hidden_states=True)
    if torch.cuda.is_available():
        model.to('cuda')

    # Get vocab indices of special tokens
    UNK_id = tokenizer.encode('[UNK]')[0]
    CLS_id = tokenizer.encode('[CLS]')[0]
    SEP_id = tokenizer.encode('[SEP]')[0]

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
    for t, t_id in zip(targets, tokenizer.encode(' '.join(targets))):
        if t_id == UNK_id:
            tokenizer.add_tokens([t])
            model.resize_token_embeddings(len(tokenizer))
            i2w[len(tokenizer) - 1] = t
        else:
            i2w[t_id] = t

    # Get sentence iterator
    sentences = PathLineSentences(corpDir)

    target_counter = {target: 0 for target in i2w}
    for sentence in sentences:
        for tok_id in tokenizer.encode(' '.join(sentence)):
            if tok_id in target_counter:
                target_counter[tok_id] += 1

    # Buffers for batched processing
    batch_input_ids, batch_targets, batch_spos = [], [], []

    # Container for usages
    usages = {
        i2w[target]: np.empty((target_count, nLayers * nDims))  # usage matrix
        for (target, target_count) in target_counter.items()
    }

    nSentences = 0
    for _ in PathLineSentences(corpDir):
        nSentences += 1

    # Get sentence iterator
    sentences = PathLineSentences(corpDir)

    # Iterate over sentences and collect representations
    nTokens = 0
    nUsages = 0
    curr_idx = {i2w[target]: 0 for target in target_counter}

    for s_id, sentence in enumerate(tqdm(sentences, total=nSentences)):

        token_ids = tokenizer.encode(' '.join(sentence))

        for spos, tok_id in enumerate(token_ids):
            nTokens += 1

            # store usage info of target words only
            if tok_id in i2w:

                # obtain context window
                context_ids, pos_in_context = get_context(token_ids, spos, contextSize)

                # add special tokens: [CLS] and [SEP]
                input_ids = [CLS_id] + context_ids + [SEP_id]

                # add usage info to buffers
                batch_input_ids.append(input_ids)
                batch_targets.append(i2w[tok_id])
                batch_spos.append(pos_in_context)

            # run model if the buffers are full or if we're at the end of the dataset
            if (len(batch_input_ids) >= bufferSize) or (s_id == nSentences - 1 and len(batch_input_ids) > 0):

                with torch.no_grad():
                    # collect list of input ids into a single batch tensor
                    input_ids_tensor = torch.tensor(batch_input_ids)
                    if torch.cuda.is_available():
                        input_ids_tensor = input_ids_tensor.to('cuda')

                    outputs = model(input_ids_tensor)

                    # extract hidden states [(bufferSize, sentLen, nDims) for l in nLayers+1]
                    if torch.cuda.is_available():
                        hidden_states = [l.detach().cpu().clone().numpy() for l in outputs[2]]
                    else:
                        hidden_states = [l.clone().numpy() for l in outputs[2]]

                # store usage tuples in a dictionary: lemma -> (vector, position)
                for b_id in np.arange(len(batch_input_ids)):
                    lemma = batch_targets[b_id]

                    # extract activations corresponding to target position
                    layers = [layer[b_id, batch_spos[b_id] + 1, :] for layer in hidden_states]
                    usage_vector = np.concatenate(layers)

                    usages[lemma][curr_idx[lemma], :] = usage_vector

                    curr_idx[lemma] += 1
                    nUsages += 1

                # finally, empty the buffers
                batch_input_ids, batch_targets, batch_spos = [], [], []

    np.savez_compressed(outPath, **usages)

    logging.info('tokens: %d' % (nTokens))
    logging.info('usages: %d' % (nUsages))
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
