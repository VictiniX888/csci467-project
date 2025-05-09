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
        input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).view(1, -1)

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
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluate_randomly(encoder, decoder, pairs, src_lang, tgt_lang, n=10):
    print(f"\nEvaluating {n} random sentence pairs...\n")
    for i in range(n):
        pair = random.choice(pairs)
        print(f"Example {i + 1}:")
        print(">", pair[0])
        print("=", pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], src_lang, tgt_lang)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def evaluate_all(encoder, decoder, pairs, src_lang, tgt_lang):
    refs = []
    hyps = []

    print("Starting full evaluation on dataset...")
    for i, (src, tgt) in enumerate(pairs):
        if i % 100 == 0:
            print(f"Evaluating pair {i+1}/{len(pairs)}...")
        decoded_words, _ = evaluate(encoder, decoder, src, src_lang, tgt_lang)
        refs.append(tgt)
        hyps.append(decoded_words)

    print("\nFinished decoding. Computing BLEU score...")
    bleu_score = compute_bleu(hyps, refs)
    print("BLEU Score:")
    print(bleu_score)
    print()

    print("Computing BERTScore...")
    P, R, F1 = compute_bertscore(hyps, refs)
    print("BERT Score:")
    print(f"P: {P.mean():.4f}, R: {R.mean():.4f}, F1: {F1.mean():.4f}")
    print()


def compute_bleu(hyps, refs):
    hyps = ["".join(sentence) for sentence in hyps]
    refs = [["".join(sentence.split()) for sentence in refs]]

    bleu = BLEU(trg_lang="en")
    score = bleu.corpus_score(hyps, refs)
    return score


def compute_bertscore(hyps, refs):
    hyps = ["".join(sentence) for sentence in hyps]
    refs = ["".join(sentence.split()) for sentence in refs]

    bert = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F1 = bert.score(hyps, refs)
    return P, R, F1
