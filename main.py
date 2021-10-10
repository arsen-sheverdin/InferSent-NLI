import argparse
# from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

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
from prepare_SNLI import *
from models import *


class MyModel(LightningModule):
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

    def on_epoch_end(self):
        #         print(trainer.optimizers[0].param_groups[0].keys())
        print('HOOK')
        dic = self.trainer.optimizers[0].param_groups[0]
        lr = dic['lr']
        print(lr)

        self.log('learning_rate', lr)

        # early stopping
        if lr < 1e-5:
            raise KeyboardInterrupt

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

def train(args):
    print(args.device)
    s = SNLI(args)
    train_iter, dev_iter, test_iter, text, label = s.get_iterators()

    model = MyModel(args, text, train_iter, dev_iter, test_iter)
    logger = TensorBoardLogger('tb_logs', name='my_model')

    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # early_stopping = EarlyStopping('val_loss')
    trainer = Trainer(
        logger=logger,
        max_epochs=40,
        val_check_interval=0.25,
        # limit_train_batches=0.01,
        # limit_val_batches=0.1
    )

    trainer.fit(model)
    trainer.test(test_dataloaders=test_iter)



def main():
    parser = argparse.ArgumentParser()
    #Basic ARGs
    parser.add_argument('--model_name', type=str, default='AWE',
                        help='The name of the model to experiment on: AWE, LSTM, BiLSTM, BiLSTM_pooling')
    parser.add_argument('--device', type=str, default= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='Device in which we perform the operations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed')
    parser.add_argument('--checkpoint_path', type = str, default = 'tb_logs',
                        help = 'Directory for saving Experiments')
    #TODO: change to
    parser.add_argument('--tokenize', type=bool, default=False,
                        help='tokenization for SNLI class')
    args = parser.parse_args()
    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(torch.backends.cudnn.enabled )


    train(args)

if __name__ == "__main__":
    main()