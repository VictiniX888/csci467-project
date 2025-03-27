import random

import torch

from constants import EOS_TOKEN
from process_data import tokenize_sentence


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tokens = tokenize_sentence(sentence, input_lang)
        input_tokens.append(EOS_TOKEN)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).view(1, -1)

        encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden, encoder_cell
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append("<EOS>")
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
