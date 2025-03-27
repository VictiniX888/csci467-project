import re

import numpy as np
import torch

from constants import EOS_TOKEN, MAX_LENGTH


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.max_length = 0

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
        if len(sentence) > self.max_length:
            self.max_length = len(sentence)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_zh_data(fname):
    with open(fname, encoding="utf-8") as f:
        lines = f.read().splitlines()

    return lines


def normalize_sentence(sentence):
    # Removes punctuation from sentences
    # Works for Chinese too
    return re.sub(r"[^\w\s]", "", sentence)


def load_sentence_pairs(src_fname, tgt_fname):
    src = read_zh_data(src_fname)
    tgt = read_zh_data(tgt_fname)
    assert len(src) == len(tgt)

    src = [normalize_sentence(sentence) for sentence in src]
    tgt = [normalize_sentence(sentence) for sentence in tgt]

    return list(zip(src, tgt))


def filter_sentence_pairs(pairs):
    return list(
        filter(
            lambda p: len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH,
            pairs,
        )
    )


def tokenize_sentence(sentence, lang):
    return [lang.word2index[word] for word in sentence.split()]


def get_dataloader(src_fname, tgt_fname, batch_size):
    pairs = load_sentence_pairs(src_fname, tgt_fname)
    print("Original pairs:", len(pairs))
    pairs = filter_sentence_pairs(pairs)
    print("Filtered pairs:", len(pairs))
    print()

    # Further filtering for debugging
    # pairs = pairs[:1000]

    src_lang = Lang("zh_old")
    tgt_lang = Lang("zh_new")

    for src_sentence, tgt_sentence in pairs:
        src_lang.add_sentence(src_sentence)
        tgt_lang.add_sentence(tgt_sentence)

    n = len(pairs)
    max_length = max(src_lang.max_length, tgt_lang.max_length)

    src_tokens = np.zeros((n, max_length), dtype=np.int32)
    tgt_tokens = np.zeros((n, max_length), dtype=np.int32)

    for i, (src_sentence, tgt_sentence) in enumerate(pairs):
        src_sentence_tokens = tokenize_sentence(src_sentence, src_lang)
        tgt_sentence_tokens = tokenize_sentence(tgt_sentence, tgt_lang)
        src_sentence_tokens.append(EOS_TOKEN)
        tgt_sentence_tokens.append(EOS_TOKEN)
        src_tokens[i, : len(src_sentence_tokens)] = src_sentence_tokens
        tgt_tokens[i, : len(tgt_sentence_tokens)] = tgt_sentence_tokens

    assert np.all(src_tokens >= 0)
    assert np.all(src_tokens < src_lang.n_words)
    assert np.all(tgt_tokens >= 0)
    assert np.all(tgt_tokens < tgt_lang.n_words)
    print("All input tensors validated")

    train_data = torch.utils.data.TensorDataset(
        torch.LongTensor(src_tokens), torch.LongTensor(tgt_tokens)
    )

    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )
    return src_lang, tgt_lang, train_dataloader, pairs
