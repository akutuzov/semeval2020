import pickle
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
        collect.py [--context=128 --buffer=128] <modelConfig> <corpDir> <testSet> <outPath>

    Arguments:
        <modelConfig> = path to file with model name, number of layers, and layer dimensionality (space-separated)    
        <corpDir> = path to corpus or corpus directory (iterates through files)
        <testSet> = path to file with one target per line
        <outPath> = output path for usage matrices and snippets

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
    with open(testSet, 'r', encoding='utf-8') as f_in:
        targets = [line.strip() for line in f_in]

    # Store vocabulary indices of target words
    i2w = {}
    for t, t_id in zip(targets, tokenizer.encode(' '.join(targets))):
        if t_id == UNK_id:
            tokenizer.add_tokens([t])
            model.resize_token_embeddings(len(tokenizer))
            i2w[len(tokenizer) - 1] = t
        else:
            i2w[t_id] = t

    # Buffers for batched processing
    batch_input_ids, batch_targets, batch_spos, batch_snippets = [], [], [], []

    # Container for usages
    usages = {
        target: (
            np.empty((0, nLayers * nDims)),  # usage matrix
            [],                              # list of snippets
            []                               # list of target's positions in snippets
        )
        for target in targets
    }

    nSentences = 0
    for _ in PathLineSentences(corpDir):
        nSentences += 1

    # Get sentence iterator
    sentences = PathLineSentences(corpDir)

    # Iterate over sentences and collect representations
    nTokens = 0
    nUsages = 0
    for s_id, sentence in enumerate(tqdm(sentences, total=nSentences)):

        token_ids = tokenizer.encode(' '.join(sentence))
        # token_ids = tokenizer.encode(sentence)

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
                batch_snippets.append(context_ids)

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

                # store usage tuples in a dictionary: lemma -> (vector, snippet, position, decade)
                for b_id in np.arange(len(batch_input_ids)):
                    lemma = batch_targets[b_id]

                    # retrieve usages stored for this lemma
                    usage_matrix, snippet_list, spos_list = usages[lemma]

                    # extract activations corresponding to target position
                    layers = [layer[b_id, batch_spos[b_id] + 1, :] for layer in hidden_states]
                    usage_vector = np.concatenate(layers)

                    # add new usage
                    usages[lemma] = (
                        np.vstack([usage_matrix, usage_vector]),
                        snippet_list + [batch_snippets[b_id]],
                        spos_list + [batch_spos[b_id]]
                    )
                    nUsages += 1

                # finally, empty the buffers
                batch_input_ids, batch_targets, batch_spos, batch_snippets = [], [], [], []

    # Write usage matrices to disk
    with open(outPath, 'wb') as f_out:
        pickle.dump(usages, file=f_out)

    logging.info('tokens: %d' % (nTokens))
    logging.info('usages: %d' % (nUsages))
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
