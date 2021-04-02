import os

from transformers import BertTokenizer, BertForMaskedLM
 # AutoConfig


if __name__ == '__main__':

    lang = 'swedish'
    model_name_or_path = 'af-ai-center/bert-base-swedish-uncased'
    use_fast = False
    lower = False
    all_forms = False
    out_path = '/Users/mario/code/semeval2020/code/bert/expanded/{}'.format(lang)
    targets_path = '/Volumes/Disk1/SemEval/finetuning_corpora/{}/targets/target_forms.csv'.format(lang)

    # Load targets
    form2target = {}
    with open(targets_path, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            entries = line.split(',')
            target, forms = entries[0], entries[1:]
            for form in forms:
                if lower:
                    form = form.lower()
                form2target[form] = target

    if all_forms:
        targets = list(set(form2target.keys()) | set(form2target.values()))
    else:
        targets = list(form2target.values())

    print('=' * 80)
    print('targets:', targets)
    print(form2target)
    print('=' * 80)

    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path,
        # cache_dir=model_args.cache_dir,
        use_fast=use_fast,
        never_split=targets,
        do_lower_case=lower
    )
    print('Tokenizer loaded.')

    model = BertForMaskedLM.from_pretrained(model_name_or_path)
    print('Model loaded.')

    tokenizer.save_pretrained(out_path)


    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    print(targets_ids)
    assert len(targets) == len(targets_ids)

    with open(os.path.join(out_path, 'vocab.txt'), 'a') as f:
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

    model.save_pretrained(out_path)