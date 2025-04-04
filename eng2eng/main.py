from preprocessing import get_dataloader
from seq2seq import Encoder, Decoder, RNNEncoder, RNNDecoder
from training import train
from evaluation import evaluate_randomly, evaluate_all, evaluate
from constants import device, BASELINE
import pdb
import torch

inLang, outLang, dataloader, pairs = get_dataloader("train.modern.txt", "train.original.txt", 256)
if BASELINE:
        encoder = RNNEncoder(inLang.n_words, 128, 128).to(device)
        decoder = RNNDecoder(
            outLang.n_words, 128, 128, inLang.max_length
        ).to(device)
else:
    encoder = Encoder(inLang.n_words, 128, 128).to(
        device
    )
    decoder = Decoder(
        outLang.n_words, 128, 128, inLang.max_length
    ).to(device)

train(dataloader, encoder, decoder, 40, 0.05, 1,1)

encoder.eval()
decoder.eval()
evaluate_randomly(encoder, decoder, pairs, inLang, outLang)



