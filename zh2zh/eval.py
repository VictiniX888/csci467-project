import random

import torch
from sacrebleu.metrics import BLEU
from bert_score import BERTScorer

from constants import EOS_TOKEN, BASELINE, device
from process_data import tokenize_sentence


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tokens = tokenize_sentence(sentence, input_lang)
        input_tokens.append(EOS_TOKEN)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).view(
            1, -1
        )

        if BASELINE:
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)
            decoder_attn = None
        else:
            encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(
                encoder_outputs, encoder_hidden, encoder_cell
            )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                # decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluate_randomly(encoder, decoder, pairs, src_lang, tgt_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], src_lang, tgt_lang)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def evaluate_all(encoder, decoder, pairs, src_lang, tgt_lang):
    refs = [tgt for _, tgt in pairs]
    hyps = [evaluate(encoder, decoder, src, src_lang, tgt_lang)[0] for src, _ in pairs]

    bleu_score = compute_bleu(hyps, refs)
    print("BLEU Score:")
    print(bleu_score)
    print()

    P, R, F1 = compute_bertscore(hyps, refs)
    print("BERT Score:")
    print(f"P: {P.mean()}, R: {R.mean()}, F1: {F1.mean()}")
    print()


def compute_bleu(hyps, refs):
    # Turn sentence vectors back into whole sentences
    hyps = ["".join(sentence) for sentence in hyps]
    refs = [["".join(sentence.split()) for sentence in refs]]

    print(hyps)
    print(refs)

    bleu = BLEU(trg_lang="zh")
    score = bleu.corpus_score(hyps, refs)
    return score


def compute_bertscore(hyps, refs):
    # Turn sentence vectors back into whole sentences
    hyps = ["".join(sentence) for sentence in hyps]
    refs = ["".join(sentence.split()) for sentence in refs]

    bert = BERTScorer(lang="zh", rescale_with_baseline=True)
    P, R, F1 = bert.score(hyps, refs)
    return P, R, F1
