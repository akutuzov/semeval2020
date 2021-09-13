import argparse
import json
import os
from transformers import BertTokenizer, BertForMaskedLM


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--targets_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument("--lang",  type=str, required=True)
    parser.add_argument('--use_fast_tokenizer', action='store_true')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--all_target_forms', action='store_true')

    args = parser.parse_args()

    # Load targets
    form2target = {}
    if args.targets_path.endswith('.json'):
        with open(args.targets_path, 'r', encoding='utf-8') as f_in:
            for line in f_in.readlines():
                line = line.strip()
                entries = line.split(',')
                target, forms = entries[0], entries[1:]
                for form in forms:
                    if args.do_lower_case:
                        form = form.lower()
                    form2target[form] = target
    else:
        if args.all_target_forms:
            raise NotImplementedError()
        with open(args.targets_path, 'r', encoding='utf-8') as f_in:
            form2target = {w: w for w in json.load(f_in) if type(w) == str}


    if args.all_target_forms:
        targets = list(form2target.keys())  # list of word forms contains the target lemma
    else:
        targets = list(form2target.values())

    print('=' * 80)
    print('targets:', targets)
    print(form2target)
    print('=' * 80)

    tokenizer = BertTokenizer.from_pretrained(
        args.model_name,
        # cache_dir=model_args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        never_split=targets,
        do_lower_case=args.do_lower_case
    )
    print('Tokenizer loaded.')

    model = BertForMaskedLM.from_pretrained(args.model_name)
    print('Model loaded.')

    tokenizer.save_pretrained(args.output_path)


    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    print(targets_ids)
    assert len(targets) == len(targets_ids)

    with open(os.path.join(args.output_path, 'vocab.txt'), 'a') as f:
        words_added = []
        for t, t_id in zip(targets, targets_ids):
            print(t, t_id)

            if tokenizer.do_lower_case:
                t = t.lower()

            if t in tokenizer.added_tokens_encoder:
                print('{} in tokenizer.added_tokens_encoder'.format(t))
                continue

            # assert len(t_id) == 1  # because of never_split list

            # print('len(t_id) > 1 ', len(t_id) > 1)
            # print('len(t_id) == 1 and t_id[0] == tokenizer.unk_token_id ', len(t_id) == 1 and t_id[0] == tokenizer.unk_token_id)

            if len(t_id) > 1 or (len(t_id) == 1 and t_id[0] == tokenizer.unk_token_id):
                if tokenizer.add_tokens([t]):
                    model.resize_token_embeddings(len(tokenizer))
                    f.write(t + '\n')
                    words_added.append(t)
                else:
                    print('Word not properly added to tokenizer:', t, tokenizer.tokenize(t))

    print("\nTarget words added to the vocabulary: {}.\n".format(', '.join(words_added)))

    model.save_pretrained(args.output_path)