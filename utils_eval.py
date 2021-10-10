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
print(device)


from collections import Counter

class AWE(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(AWE, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)

    def forward(self, x):
        embedded = self.embedding(x)
        awe = torch.mean(embedded, axis=1)

        #         outputs, _ = self.rnn(embedded, (h0, c0))
        #         prediction = self.fc_out(outputs[-1, :, :])

        return awe

class AverageEmbeddings(LightningModule):
    def __init__(self, config, text, train_iter, dev_iter, test_iter ):
        super().__init__()

        self.nli_net = NLINet(config, text).to(device = device)

        weight = torch.FloatTensor(3).fill_(1)
        self.loss_function = nn.CrossEntropyLoss(weight=weight)
        self.valid_losses = []
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter

    def forward(self, x):
        # X is vector of shape (batch, input, )
        # need to be permuted because by default X is batch first
        return self.nli_net(x[0], x[1])

    # def on_epoch_end(self):
    #     #         print(trainer.optimizers[0].param_groups[0].keys())
    #     print('HOOK')
    #     dic = self.trainer.optimizers[0].param_groups[0]
    #     lr = dic['lr']
    #     print(lr)

    #     self.log('learning_rate', lr)

    #     # early stopping
    #     if lr < 1e-5:
    #         raise KeyboardInterrupt

    def training_step(self, batch, batch_idx):
        premise = batch.premise[0].to(device=device)
        hypothesis = batch.hypothesis[0].to(device=device)
        targets = batch.label.to(device=device)
        # TODO: Accuracies in all of the,loops

        y_hat = self.nli_net(premise, hypothesis)

        loss = self.loss_function(y_hat, targets.type_as(y_hat).long())

        predict = torch.argmax(y_hat, axis=1)
        acc = FM.accuracy(predict, targets)

        metrics_train = {'train_acc': acc, 'train_loss': loss}
        self.log('train_loss', loss, on_step=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_epoch', loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        premise = batch.premise[0].to(device=device)
        hypothesis = batch.hypothesis[0].to(device=device)
        targets = batch.label.to(device=device)

        y_hat = self.nli_net(premise, hypothesis)
        loss = self.loss_function(y_hat, targets.type_as(y_hat).long())

        predict = torch.argmax(y_hat, axis=1)
        acc = FM.accuracy(predict, targets)
        metrics_val = {'val_acc': acc, 'val_loss': loss}

        self.log_dict(metrics_val)
        self.log('val_acc_on_EPOCH', acc, on_epoch=True)

        return metrics_val

    def test_step(self, batch, batch_idx):
        premise = batch.premise[0].to(device=device)
        hypothesis = batch.hypothesis[0].to(device=device)
        targets = batch.label.to(device=device)

        y_hat = self.nli_net(premise, hypothesis)
        loss = self.loss_function(y_hat, targets.type_as(y_hat).long())
        predict = torch.argmax(y_hat, axis=1)

        predict = torch.argmax(y_hat, axis=1)
        acc = FM.accuracy(predict, targets)

        metrics_test = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics_test, prog_bar=True, logger=True)

class Recurrent(LightningModule):
    def __init__(self, config, text, train_iter, dev_iter, test_iter):
        super().__init__()

        self.nli_net = NLINet(config, text)

        weight = torch.FloatTensor(3).fill_(1)
        self.loss_function = nn.CrossEntropyLoss(weight=weight)
        self.valid_losses = []
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter

    def forward(self, x):
        # X is  sentence 1 and 2
        return self.nli_net(x[0], x[1])

    # def on_epoch_end(self):
        #         print(trainer.optimizers[0].param_groups[0].keys())
        # print('HOOK')
        # dic = self.trainer.optimizers[0].param_groups[0]
        # lr = dic['lr']
        # print(lr)

        # self.log('learning_rate', lr)

        # # early stopping
        # if lr < 1e-5:
        #     raise KeyboardInterrupt

    def training_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        targets = batch.label
        # TODO: Accuracies in all of the,loops

        y_hat = self.nli_net(premise, hypothesis)

        loss = self.loss_function(y_hat, targets.type_as(y_hat).long())

        predict = torch.argmax(y_hat, axis=1)
        acc = FM.accuracy(predict, targets)

        metrics_train = {'train_acc': acc, 'train_loss': loss}
        self.log('train_loss', loss, on_step=True)

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_epoch', loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        targets = batch.label

        y_hat = self.nli_net(premise, hypothesis)
        loss = self.loss_function(y_hat, targets.type_as(y_hat).long())

        predict = torch.argmax(y_hat, axis=1)
        acc = FM.accuracy(predict, targets)
        metrics_val = {'val_acc': acc, 'val_loss': loss}

        self.log_dict(metrics_val)
        self.log('val_acc_on_EPOCH', acc, on_epoch=True)

        return metrics_val

    def test_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        targets = batch.label

        y_hat = self.nli_net(premise, hypothesis)
        loss = self.loss_function(y_hat, targets.type_as(y_hat).long())

        predict = torch.argmax(y_hat, axis=1)
        acc = FM.accuracy(predict, targets)

        metrics_test = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics_test, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # TODO: Learning_RATE
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        #
        scheduler1 = ReduceLROnPlateau(optimizer, mode='max', factor=0.2,
                                       verbose=True, patience=2)

        return ({'optimizer': optimizer, "lr_scheduler": scheduler1, "monitor": "val_acc"})

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.dev_iter

    def test_dataloader(self):
        return self.test_iter


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
            self.encoder = AWE(37179, 300, self.hidden, 1)
            self.inputdim = 4 * 300
        if config.model_name == 'LSTM':
            self.encoder = RNN_LSTM(37179, self.enc_lstm_dim, self.hidden, 1)
            self.inputdim = 4 * 2048
        if config.model_name == 'BiLSTM':
            self.encoder = RNN_BiLSTM(37179,  self.enc_lstm_dim, self.hidden, 1)
            self.inputdim = 4 * 4096

        pretrained_embeddings = torch.cat((text.vocab.vectors, torch.zeros(4, 300)), 0)
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
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u - v), u * v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb
