import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import SOS_TOKEN, device


# Adapted from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# and https://nthu-datalab.github.io/ml/labs/12-1_Seq2Seq-Learning_Neural-Machine-Translation/12-1_Seq2Seq-Learning_Neural-Machine-Translation.html


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_hidden_state, decoder_hidden_state):
        attn_scores = self.V(
            torch.tanh(self.W1(decoder_hidden_state) + self.W2(encoder_hidden_state))
        )
        attn_scores = attn_scores.squeeze(2).unsqueeze(1)

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_hidden_state)

        return context, attn_weights


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hn, cn) = self.lstm(embedded)
        return output, hn, cn


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, max_length):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(2 * embedding_dim, lstm_dim, batch_first=True)
        self.out = nn.Linear(lstm_dim, vocab_size)
        self.attention = AttentionLayer(embedding_dim)

        self.max_length = max_length

    def forward(
        self, encoder_outputs, encoder_hidden, encoder_cell, target_tensor=None
    ):
        B = encoder_outputs.size(0)
        decoder_input = torch.empty(B, 1, dtype=torch.long, device=device).fill_(
            SOS_TOKEN
        )
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        decoder_outputs = []
        # decoder_cells = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, decoder_cell, attn_weights = (
                self.forward_step(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                )
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, decoder_hidden, decoder_cell, encoder_hidden):
        embedded = self.embedding(input)
        decoder_hidden_p = decoder_hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(encoder_hidden, decoder_hidden_p)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hn, cn) = self.lstm(lstm_input, (decoder_hidden, decoder_cell))
        output = self.out(output)
        return output, hn, cn, attn_weights


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_dim):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, rnn_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hn = self.rnn(embedded)
        return output, hn


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_dim, max_length):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, rnn_dim, batch_first=True)
        self.out = nn.Linear(rnn_dim, vocab_size)

        self.max_length = max_length

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        B = encoder_outputs.size(0)
        decoder_input = torch.empty(B, 1, dtype=torch.long, device=device).fill_(
            SOS_TOKEN
        )
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden

    def forward_step(self, input, decoder_hidden):
        embedded = self.embedding(input)
        output, hn = self.rnn(embedded, decoder_hidden)
        output = self.out(output)
        return output, hn
