import re
import torch
import numpy as np
from constants import SOS_token, EOS_token, MAX_LENGTH, device


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.max_length = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        if len(sentence) > self.max_length:
            self.max_length = len(sentence)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readData(name):
    return open(name).read().splitlines()


def pairSentences(source, target):
    s = readData(source)
    t = readData(target)
    s = [normalizeString(sent) for sent in s]
    t = [normalizeString(sent) for sent in t]
    return list(zip(s,t))


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def tokenize_sentence(sentence, lang):
    return [lang.word2index[word] for word in sentence.split()]


def get_dataloader(source, target, batch_size):
    pairs = pairSentences(source, target)
    print("Original pairs:", len(pairs))
    pairs = filterPairs(pairs)
    print("Filtered pairs:", len(pairs))
    print()

    # Further filtering for debugging
    # pairs = pairs[:1000]

    src_lang = Lang("eng_new")
    tgt_lang = Lang("eng_old")

    for src_sentence, tgt_sentence in pairs:
        src_lang.addSentence(src_sentence)
        tgt_lang.addSentence(tgt_sentence)

    n = len(pairs)
    max_length = max(src_lang.max_length, tgt_lang.max_length)

    src_tokens = np.zeros((n, max_length), dtype=np.int32)
    tgt_tokens = np.zeros((n, max_length), dtype=np.int32)

    for i, (src_sentence, tgt_sentence) in enumerate(pairs):
        src_sentence_tokens = tokenize_sentence(src_sentence, src_lang)
        tgt_sentence_tokens = tokenize_sentence(tgt_sentence, tgt_lang)
        src_sentence_tokens.append(EOS_token)
        tgt_sentence_tokens.append(EOS_token)
        src_tokens[i, : len(src_sentence_tokens)] = src_sentence_tokens
        tgt_tokens[i, : len(tgt_sentence_tokens)] = tgt_sentence_tokens

    train_data = torch.utils.data.TensorDataset(
        torch.LongTensor(src_tokens).to(device), torch.LongTensor(tgt_tokens).to(device)
    )

    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )
    return src_lang, tgt_lang, train_dataloader, pairs

