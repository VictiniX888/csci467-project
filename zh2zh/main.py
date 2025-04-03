import torch
from process_data import get_dataloader
from model import Seq2SeqEncoder, Seq2SeqDecoder
from train import train
from eval import evaluate_randomly
from constants import (
    DEV_SRC_FNAME,
    DEV_TGT_FNAME,
    ENCODER_FNAME,
    DECODER_FNAME,
    TRAIN_SRC_FNAME,
    TRAIN_TGT_FNAME,
)


def train_models():
    hidden_size = 128
    batch_size = 32

    input_lang, output_lang, train_dataloader, pairs = get_dataloader(
        TRAIN_SRC_FNAME, TRAIN_TGT_FNAME, batch_size
    )

    print("Input words:", input_lang.n_words)
    print("Input dict len:", len(input_lang.index2word))
    print("Input dict:")
    for i in range(5):
        print(i, input_lang.index2word[i])
    print()

    print("Output words:", output_lang.n_words)
    print("Output dict len:", len(output_lang.index2word))
    print("Output dict:")
    for i in range(5):
        print(i, output_lang.index2word[i])
    print()

    encoder = Seq2SeqEncoder(input_lang.n_words, hidden_size, hidden_size)
    decoder = Seq2SeqDecoder(
        output_lang.n_words, hidden_size, hidden_size, output_lang.max_length
    )

    train(train_dataloader, encoder, decoder, 80, print_every=1, plot_every=1)

    torch.save(encoder.state_dict(), ENCODER_FNAME)
    torch.save(decoder.state_dict(), DECODER_FNAME)


def eval_models():
    hidden_size = 128
    batch_size = 32

    input_lang, output_lang, train_dataloader, pairs = get_dataloader(
        TRAIN_SRC_FNAME, TRAIN_TGT_FNAME, batch_size
    )

    print("Input words:", input_lang.n_words)
    print("Input dict len:", len(input_lang.index2word))
    print("Input dict:")
    for i in range(5):
        print(i, input_lang.index2word[i])
    print()

    print("Output words:", output_lang.n_words)
    print("Output dict len:", len(output_lang.index2word))
    print("Output dict:")
    for i in range(5):
        print(i, output_lang.index2word[i])
    print()

    encoder = Seq2SeqEncoder(input_lang.n_words, hidden_size, hidden_size)
    encoder.load_state_dict(torch.load(ENCODER_FNAME, weights_only=True))

    decoder = Seq2SeqDecoder(
        output_lang.n_words, hidden_size, hidden_size, output_lang.max_length
    )
    decoder.load_state_dict(torch.load(DECODER_FNAME, weights_only=True))

    encoder.eval()
    decoder.eval()
    evaluate_randomly(encoder, decoder, pairs, input_lang, output_lang)


if __name__ == "__main__":
    # train_models()
    eval_models()
