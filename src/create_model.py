"""
A script to save a bentoml model
"""
import pathlib
import random
import re
import unicodedata

import bentoml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator

CURRENT_PATH = pathlib.Path(__file__).resolve().parent

MAX_LEN = 10
# Required for model
MAX_SEQUENCE_LENGTH = MAX_LEN + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN_INDEX = 1


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, bidirectional=False, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        output = self.dropout(output)
        return output, hidden


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_size,
        bidirectional=False,
        dropout=0.0,
        fill_val=SOS_TOKEN_INDEX,
    ):
        super().__init__()
        self.fill_val = fill_val
        self.embedding = nn.Embedding(output_size, hidden_dim)
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        fc_in_features = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_features, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, hidden_state, target_tensor=None):
        batch_size = encoder_outputs.shape[0]
        input = torch.zeros(batch_size, 1, dtype=torch.long, device=device).fill_(
            self.fill_val
        )
        outputs = []

        for i in range(MAX_SEQUENCE_LENGTH):
            output, _ = self.forward_step(input, hidden_state)
            # output is of shape: [batch_size, 1, output_size], where output_size is
            # num. of unique words in target language
            outputs.append(output)

            if target_tensor is not None:
                # If teacher forcing:
                # Use the target tensor's values as the next input, converting them to
                # the same shape as the original input
                input = target_tensor[:, i].unsqueeze(1)
            else:
                # If not teacher forcing:
                # Take the topk of the output and use it as the next input (the topk
                # will be of size [batch_size, 1, 1])
                _, top_indexes = output.topk(1)
                input = top_indexes.squeeze(-1)

        # Concatenate all of the outputs along dimension 1, creating the complete
        # sequence from individual parts for each input
        # (size [batch_size, MAX_SEQUENCE_LENGTH, output_size])
        outputs = torch.cat(outputs, dim=1)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def forward_step(self, input, hidden_state):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden_state = self.rnn(output, hidden_state)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden_state


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        bidirectional=True,
        dropout=0.0,
        fill_val=SOS_TOKEN_INDEX,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_size, hidden_dim, bidirectional=bidirectional, dropout=dropout
        )
        self.decoder = Decoder(
            hidden_dim,
            output_size,
            bidirectional=bidirectional,
            dropout=dropout,
            fill_val=fill_val,
        )

    def forward(self, input, target_tensor=None):
        encoder_output, encoder_hidden = self.encoder(input)
        decoder_output = self.decoder(encoder_output, encoder_hidden, target_tensor)
        return decoder_output


# Required to build vocabs
DATASET_PATH = CURRENT_PATH / "data/eng-fra.txt"
DATASET_USAGE = 1
PAIR_DELIMETER = "\t"

UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"


def read_pairs():
    with open(DATASET_PATH, "r", encoding="utf8") as f:
        pairs = f.read().strip().split("\n")
        random.shuffle(pairs)
        pairs = pairs[: int(DATASET_USAGE * len(pairs))]
        return pairs


# Remove any accented characters and non-ASCII symbols
# From: https://stackoverflow.com/a/7782177
def normalize_text(text):
    return str(
        unicodedata.normalize("NFKD", text.strip().lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def remove_special_chars(text):
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z!?]+", " ", text)
    return text.strip()


def preprocess_text(text):
    text = normalize_text(text)
    return remove_special_chars(text)


def read_normalized_pairs():
    pairs = read_pairs()
    for index, pair in enumerate(pairs):
        eng, fra = pair.split(PAIR_DELIMETER)
        eng = preprocess_text(eng)
        fra = preprocess_text(fra)
        pairs[index] = [eng, fra]
    filtered = [
        pair
        for pair in pairs
        if len(pair[0].split(" ")) <= MAX_LEN and len(pair[1].split(" ")) <= MAX_LEN
    ]
    return filtered


def eng_iter(pairs):
    for eng, fra in pairs:
        yield tokenize(eng)


def fra_iter(pairs):
    for eng, fra in pairs:
        yield tokenize(fra)


def build_vocab(pairs=None):
    if pairs is None:
        pairs = read_normalized_pairs()
    specials = [UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
    eng_vocab = build_vocab_from_iterator(
        eng_iter(pairs), specials=specials, special_first=True
    )
    fra_vocab = build_vocab_from_iterator(
        fra_iter(pairs), specials=specials, special_first=True
    )
    eng_vocab.set_default_index(eng_vocab[UNK_TOKEN])
    fra_vocab.set_default_index(fra_vocab[UNK_TOKEN])
    return eng_vocab, fra_vocab


# Required for translate()
def tokenize(sentence):
    return sentence.lower().split(" ")


def text_pipeline(text, vocab):
    return vocab(tokenize(text))


def prepare_input(text, vocab):
    text = text_pipeline(text, vocab)
    text += vocab([EOS_TOKEN])
    return text


def tensor_to_sentence(model_output_indexes, vocab):
    sentence = []
    for word_index in model_output_indexes:
        word = vocab.lookup_token(word_index)
        if word == EOS_TOKEN:
            break
        sentence.append(word)
    return " ".join(sentence)


# Create and save the model with bentoml

HIDDEN_DIM = 512
WEIGHTS_PATH = CURRENT_PATH / "weights/en-fr_rnn_lstm_512.pt"

eng_vocab, fra_vocab = build_vocab()
model = Seq2SeqModel(len(eng_vocab), len(fra_vocab), HIDDEN_DIM, bidirectional=True)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))


bentoml.pytorch.save_model(
    "en-fr_rnn_lstm_512",
    model,
    signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    custom_objects={
        "prepare_input": prepare_input,
        "tensor_to_sentence": tensor_to_sentence,
        "eng_vocab": eng_vocab,
        "fra_vocab": fra_vocab,
    },
)
