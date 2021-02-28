import argparse
import pickle
import torch
import time
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import normalize
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
        '--subs_path', type=str, required=True,
        help='Path to the pickle file containing substitute lists (output by substitutes.py).')
    parser.add_argument(
        '--targets_path', type=str, required=True,
        help='Path to the csv file containing target word forms.')
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Output path for pickle containing substitutes with lexical similarity values.')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='The batch size per device (GPU core / CPU).')
    parser.add_argument(
        '--ignore_decoder_bias', action='store_true',
        help="Whether to ignore the decoder's bias vector during masked word prediction")
    parser.add_argument(
        '--normalise_embeddings', action='store_true',
        help="Whether to ignore the decoder's bias vector during masked word prediction")
    parser.add_argument(
        '--local_rank', type=int, default=-1,
        help='For distributed training.')
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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name, never_split=target_forms, use_fast=False)
    model = BertForMaskedLM.from_pretrained(args.model_name, output_hidden_states=True)

    if args.ignore_decoder_bias:
        logger.warning('Ignoring bias vector for masked word prediction.')
        model.cls.predictions.decoder.bias = None

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    # Store vocabulary indices of target words
    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in target_forms]
    assert len(target_forms) == len(targets_ids)
    words_added = []
    for t, t_id in zip(target_forms, targets_ids):
        if tokenizer.do_lower_case:
            t = t.lower()
        if t in tokenizer.added_tokens_encoder:
            continue
        if len(t_id) > 1 or (len(t_id) == 1 and t_id[0] == tokenizer.unk_token_id):
            if tokenizer.add_tokens([t]):
                model.resize_token_embeddings(len(tokenizer))
                words_added.append(t)
            else:
                logger.error('Word not properly added to tokenizer:', t, tokenizer.tokenize(t))

    # check if correctly added
    for t, t_id in zip(target_forms, targets_ids):
        if len(t_id) != 1:
            print(t, t_id)
    logger.warning("\nTarget words added to the vocabulary: {}.\n".format(', '.join(words_added)))

    # assert len(t_id) == 1  # because of never_split list
    # if t_id[0] == tokenizer.unk_token_id:
    #     if tokenizer.add_tokens([t]):
    #         model.resize_token_embeddings(len(tokenizer))
    #         words_added.append(t)
    #     else:
    #         logger.error('Word not properly added to tokenizer:', t, tokenizer.tokenize(t))


    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    with open(args.subs_path, 'rb') as f_in:
        substitutes_raw = pickle.load(f_in)

    substitutes_new = {
        w: [{'candidates': [], 'logp': [], 'dot_products': []} for _ in substitutes_raw[w]]
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

    dataset = SubstitutesDataset(substitutes_raw, tokenizer, args.normalise_embeddings)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
    iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(iterator):
        model.eval()

        inputs, tgt, occurrence_idxs, candidate_tokens, tgt_embedding, logps, positions = batch
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        tgt_embedding = tgt_embedding.to(device)
        bsz = inputs['input_ids'].shape[0]

        with torch.no_grad():
            outputs = model(**inputs)  # n_sentences, max_sent_len, vocab_size

            hidden_states = outputs[1]
            last_layer = hidden_states[-1][np.arange(bsz), positions, :]  # (bsz, hdims)
            if args.normalise_embeddings:
                last_layer = normalize(last_layer, p=2)

            dot_products = torch.sum(tgt_embedding * last_layer, dim=1)  # (bsz)

            if args.normalise_embeddings:
                assert all([d <= 1.01 for d in dot_products]), 'Dot product should not exceed 1 if vectors are normalised.'

            for b_id in np.arange(bsz):
                tgt_lemma = tgt[b_id]
                occurrence_idx = occurrence_idxs[b_id]

                substitutes_new[tgt_lemma][occurrence_idx]['candidates'].append(candidate_tokens[b_id])
                substitutes_new[tgt_lemma][occurrence_idx]['logp'].append(logps[b_id])
                substitutes_new[tgt_lemma][occurrence_idx]['dot_products'].append(dot_products[b_id].item())

    iterator.close()

    with open(args.output_path, 'wb') as f_out:
        pickle.dump(substitutes_new, f_out)

    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
