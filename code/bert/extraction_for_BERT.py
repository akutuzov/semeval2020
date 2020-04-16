import re
import string
import numpy as np
import torch
import random
import os

from docopt import docopt
from nltk.corpus import stopwords
from nltk import sent_tokenize
from collections import Counter
from transformers import BertTokenizer, BertModel


# def load_coha_sentences(decade, coha_path):
#     coha_path = coha_path + str(decade)
#     print("Loading COHA sentences from", coha_path)
#     coha_files = os.listdir(coha_path)
#     sentences = []
#     for coha_file in coha_files:
#         if ".txt" in coha_file:
#             coha_filepath = coha_path + '/' + coha_file
#             try:
#                 text = open(coha_filepath, 'r').read().lower()
#             except:
#                 text = open(coha_filepath, 'rb').read().decode('utf-8').lower()
#             sentences.extend(sent_tokenize(text))
#     return sentences


def load_coha_sentences(path):
    sentences = []
    try:
        text = open(path, 'r').read().lower()
    except:
        text = open(path, 'rb').read().decode('utf-8').lower()

    sentences.extend(sent_tokenize(text))
    return sentences


def get_embedding_for_sentence(tokenized_sent):
    #print("Getting embedding for sentence")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(tokenized_sent)
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        _, _, encoded_layers = model(tokens_tensor, segments_tensors)
        batch_i = 0
        token_embeddings = []
        # For each token in the sentence...
        for token_i in range(len(tokenized_sent)):
            hidden_layers = []
            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings]
        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
        last_layer = [layer[-1] for layer in token_embeddings]
        return summed_last_4_layers


def get_embeddings_for_word(word, sentences):
    print("Getting BERT embeddings for word:", word)
    word_embeddings = []
    # valid_sentences = []
    for i, sentence in enumerate(sentences):
            marked_sent = "[CLS] " + sentence + " [SEP]"
            tokenized_sent = tokenizer.tokenize(marked_sent)
            if word in tokenized_sent and 512 > len(tokenized_sent) > 3:
                sent_embedding = get_embedding_for_sentence(tokenized_sent)
                word_indexes = list(np.where(np.array(tokenized_sent) == word)[0])
                for index in word_indexes:
                    word_embedding = np.array(sent_embedding[index])
                    word_embeddings.append(word_embedding)
                    # valid_sentences.append(sentence)
    word_embeddings = np.array(word_embeddings)
    # valid_sentences = np.array(valid_sentences)
    return word_embeddings  #, valid_sentences


def extract_vocabulary(sentences):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['also'])
    punctuation = set(string.punctuation)
    vocab=[]

    for index, sentence in enumerate(sentences):
        if index%1000==0:
            print(str(index) + '/' + str(len(sentences)))
        words = re.sub("[^\w]", " ", sentence).split()
        for word in words:
            word_clean = ''
            for elem in word:
                if elem not in punctuation and not elem.isdigit():
                    word_clean += elem
            if len(word_clean) > 1 and word_clean not in stop_words:  # delete all words with only one character
                vocab.append(word_clean)
    vocab_freq = Counter(vocab)
    return vocab_freq


def get_embeddings_for_word_oov(word, sentences, oneEmbPerSentence = True):
    print("Getting BERT embeddings for word:", word)
    # test to see if it is a divided word
    marked_sent = "[CLS] " + sentences[0] + " [SEP]"
    tokenized_sent = tokenizer.tokenize(marked_sent)
    word_embeddings = []
    sent_embedding = get_embedding_for_sentence(tokenized_sent)
    if word not in tokenized_sent:
        print("Divided word !! ")
        print(tokenized_sent)
    # prepare sentences for BERT
    for i, sentence in enumerate(sentences):
        sent_embs = []
        if i%500==0:
            print('step2: sent ' + str(i) + ' out of ' + str(len(sentences)))
        marked_sent = "[CLS] " + sentence + " [SEP]"
        tokenized_sent = tokenizer.tokenize(marked_sent)
        # get set of token embeddings
        sent_embedding = get_embedding_for_sentence(tokenized_sent)
        if word in tokenized_sent:
            word_indexes = list(np.where(np.array(tokenized_sent) == word)[0])
            for index in word_indexes:
                word_embedding = np.array(sent_embedding[index])
                sent_embs.append(word_embedding)
        # in case the word is not found because divided into byte-pair encodings:
        else:
            # in case the word is divided in more than 2 pieces:
            splitted_tokens = []
            splitted_array = np.zeros((1, 768))
            prev_token = ""
            prev_array = np.zeros((1, 768))

            for i, token_i in enumerate(tokenized_sent):
                array = sent_embedding[i]
                token_i = token_i.lower()

                # Find the word divided into several byte-pair encodings
                if token_i.startswith('##'):

                    if prev_token:
                        splitted_tokens.append(prev_token)
                        prev_token = ""
                        splitted_array = prev_array

                    splitted_tokens.append(token_i)
                    splitted_array += array

                else:
                    if splitted_tokens:
                        sarray = splitted_array/len(splitted_tokens)
                        stoken_i = "".join(splitted_tokens).replace('##', '')
                        if stoken_i.lower() == word:
                            # replace in the tokenized sentence the divided word by the full word
                            # tokenized_sent[i-len(splitted_tokens):i] = word
                            word_embedding = np.array(sarray)
                            # keep all the embs of the sentence, to select only one randomly
                            sent_embs.append(word_embedding)
                        splitted_tokens = []
                        splitted_array = np.zeros((1, 768))
                    prev_array = array
                    prev_token = token_i

        # We want a fixed amount of embeddings. So we select only one occurence per sentence, randomly.
        if oneEmbPerSentence:
            if len(sent_embs)>1:
                emb = random.choice(sent_embs)
            elif len(sent_embs) == 1:
                emb = sent_embs[0]
            else:
                emb = 0
            word_embeddings.append(np.array(emb))
        else:
            for emb in sent_embs:
                word_embeddings.append(np.array(emb))

    word_embeddings = np.array(word_embeddings)
    return word_embeddings


if __name__ == '__main__':

    args = docopt("""Collect usages like Martinc et al. (2020)

    Usage:
        extraction_for_BERT.py <model> <targets> <corpus> <outPath>

    Arguments:
        <model> = path to model or model name
        <targets> = path to target words
        <corpus> = path to corpus
        <outPath> = output path for .npz file
    """)

    path_targets = args['<targets>']
    path_corpus = args['<corpus>']
    out_path = args['<outPath>']
    bert_model = args['<model>']

    skip = ['extracellular', 'sulphate', 'assay', 'mediaeval']

    print('Read targets: {}'.format(path_targets))
    with open(path_targets, 'r', encoding='utf-8') as f:
        targets = [w.strip() for w in f.readlines() if w not in skip]

    print('{} target words'.format(len(targets)))

    print('Load sentences: {}'.format(path_corpus))
    sentences = load_coha_sentences(path_corpus)
    print('{} - {} sentences'.format(path_corpus, len(sentences)))

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
    model.eval()

    print('Collect usages')
    usages = {}
    for word in targets:
        usages[word] = get_embeddings_for_word(word, sentences)

    print('Save usages: {}'.format(out_path))
    np.savez_compressed(out_path, **usages)