import pickle
import torch
import time
import logging
import numpy as np
from tqdm import tqdm
from docopt import docopt
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForMaskedLM

logger = logging.getLogger(__name__)


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
                    embedding = occurrence['embedding'] / occurrence['embedding'].sum()
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

    # Get the arguments
    args = docopt("""Modify substitute probabilities based on lexical similarity with target.

    Usage:
        inject_lexical_similarity.py [--batch=B --localRank=R --temperature=T --normalise --ignoreBias] <modelName> <testSet> <subsPath> <outPath>

    Arguments:
        <modelName> = HuggingFace model name 
        <testSet> = path to file with one target per line
        <subsPath> = path to pickle containing substitute lists
        <outPath> = output path for substitutes with updated probabilities
    Options:
        --batch=B  The batch size [default: 64]
        --localRank=R  For distributed training [default: -1]
        --temperature=T  The temperature value for the lexical similarity calculation [default: 1.]
        --normalise  Whether to normalise the embeddings before dot product
        --ignoreBias  Whether to ignore the bias vector during masked word prediction
    """)

    modelName = args['<modelName>']
    testSet = args['<testSet>']
    subsPath = args['<subsPath>']
    outPath = args['<outPath>']
    batchSize = int(args['--batch'])
    localRank = int(args['--localRank'])
    ignore_lm_bias = bool(args['--ignoreBias'])
    normaliseEmbed = bool(args['--normalise'])
    temperature = torch.tensor(float(args['--temperature']))

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

    with open(subsPath, 'rb') as f_in:
        substitutes_raw = pickle.load(f_in)

    substitutes_new = {
        w: [{'candidates': [], 'logp': []} for _ in substitutes_raw[w]]
        for w in substitutes_raw
    }

    def collate(batch):
        return [
            {'input_ids': torch.cat([item[0]['input_ids'].unsqueeze(0) for item in batch], dim=0),
             'attention_mask': torch.cat([item[0]['attention_mask'].unsqueeze(0) for item in batch], dim=0)},
            [item[1] for item in batch],  # target
            [item[2] for item in batch],  # occurrence_idx
            [item[3] for item in batch],  # candidate_token
            torch.cat([torch.as_tensor(item[4]).unsqueeze(0) for item in batch], dim=0),  # embedding
            [item[5] for item in batch],  # logp
            [item[6] for item in batch]   #position
        ]

    dataset = SubstitutesDataset(substitutes_raw, tokenizer, normaliseEmbed)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batchSize, collate_fn=collate)
    iterator = tqdm(dataloader, desc="Iteration", disable=localRank not in [-1, 0])

    for step, batch in enumerate(iterator):
        model.eval()

        inputs, tgt, occurrence_idxs, candidate_tokens, tgt_embedding, logps, positions = batch
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        bsz = inputs['input_ids'].shape[0]

        with torch.no_grad():
            outputs = model(**inputs)  # n_sentences, max_sent_len, vocab_size

            hidden_states = outputs[1]
            last_layer = hidden_states[-1][np.arange(bsz), positions, :]  # (bsz, hdims)
            if normaliseEmbed:
                last_layer /= last_layer.sum()

            dot_products = torch.sum(tgt_embedding * last_layer, dim=1)  # (bsz)
            dot_products = dot_products / temperature

            for b_id in np.arange(bsz):
                tgt_lemma = tgt[b_id]
                occurrence_idx = occurrence_idxs[b_id]

                substitutes_new[tgt_lemma][occurrence_idx]['candidates'].append(candidate_tokens[b_id])
                substitutes_new[tgt_lemma][occurrence_idx]['logp'].append(logps[b_id] + dot_products[b_id].item())

    iterator.close()

    for lemma in substitutes_new:
        for occurrence in substitutes_new[lemma]:
            indices = np.argsort(occurrence['logp'])[::-1]
            occurrence['logp'] = [occurrence['logp'][j] for j in indices]
            occurrence['candidates'] = [occurrence['candidates'][j] for j in indices]

    with open(outPath, 'wb') as f_out:
        pickle.dump(substitutes_new, f_out)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
