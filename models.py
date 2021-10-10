import sys
print(sys.executable)

import sklearn
sklearn.__version__

import torch.nn as nn
import torch.optim as optim

import nltk
from nltk import word_tokenize
nltk.download('punkt')

import dill

print('next')
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch
import torchtext.vocab as vocab
import time


from torchtext.data import Field
from torchtext.datasets import IMDB
from torchtext.data import BucketIterator
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import tensorboard
from pytorch_lightning.metrics import functional as FM
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
seed_everything(42)
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence




class AWE(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(AWE, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)

    def forward(self, x):
        embedded = self.embedding(x)
        awe = torch.mean(embedded, axis=1)

        return awe


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = 2048
        self.num_layers = 1
        embed_size = 300

        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=1)
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, bidirectional=False)

    def forward(self, x):
        src, src_len = x[0], x[1]
        src = self.embedding(src)

        src = pack_padded_sequence(src, src_len, batch_first=True, enforce_sorted=False)

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x[0].shape[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers, x[0].shape[0], self.hidden_size)
        output, (hn, cn) = self.rnn(src, (h0, c0))

        return hn[0]


class RNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_BiLSTM, self).__init__()
        self.hidden_size = 2048
        self.num_layers = 1
        embed_size = 300

        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=1)
        self.rnn = nn.LSTM(input_size=embed_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=True)



    def forward(self, x):
        # Set initial states

        src, src_len = x[0], x[1]
        src = self.embedding(src)

        src = pack_padded_sequence(src, src_len, batch_first=True, enforce_sorted=False)

        h0 = torch.zeros(self.num_layers * 2, x[0].shape[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x[0].shape[0], self.hidden_size)

        # Forward propagate LSTM
        output, (hn, cn) = self.rnn(src, (h0, c0))

        forward_state = hn[-2, :, :]
        backward_state = hn[-1, :, :]

        representation = torch.cat((forward_state, backward_state), dim=1)

        return representation

class NLINet(nn.Module):
    def __init__(self, config, text):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = True
        self.n_classes = 3
        self.enc_lstm_dim = 300
        self.dpout_fc = 0.1
        self.fc_dim = 512
        self.n_classes = 3
        self.hidden = config.hidden_dim #default 2048

        # TODO: CHANGABLE ENCODERS
        if config.model_name == 'AWE':
            self.encoder = AWE(len(text.vocab), 300, self.hidden, 1)
        if config.model_name == 'LSTM':
            self.encoder = RNN_LSTM(len(text.vocab), self.enc_lstm_dim, self.hidden, 1)
            self.inputdim = 4 * self.hidden
        if config.model_name ==   'BiLSTM':
            self.encoder = RNN_BiLSTM(len(text.vocab),  self.enc_lstm_dim, self.hidden, 1)
            self.inputdim = 4 * 4096

        pretrained_embeddings = text.vocab.vectors
        self.encoder.embedding.weight.data.copy_(pretrained_embeddings)

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
            )

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u - v), u * v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb