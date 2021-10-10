import sys
print(sys.executable)

import sklearn
sklearn.__version__

import nltk
from nltk import word_tokenize
nltk.download('punkt')

print('next')
from torchtext import data
from torchtext import datasets
import torch
import torchtext.vocab as vocab
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class SNLI():
    def __init__(self, args):
        start = time.time()

        #TODO: turn on tokenizer
        if args.tokenize == False:
            self.TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
        else:
            self.TEXT = data.Field(batch_first=True, include_lengths=True, tokenize=word_tokenize, lower=True)

        self.LABEL = data.Field(sequential=False, unk_token=None)

        # dataset

        self.train, self.dev, self.test = datasets.SNLI.splits(text_field=self.TEXT,
                                                               label_field=self.LABEL,
                                                               root='dataset'
                                                               )
        # train='dataset/snli/snli_1.0_train.jsonl',
        # validation='dataset/snli/snli_1.0_dev.jsonl',
        # test='dataset/snli/snli_1.0_test.jsonl',
        # )
        #
        print('split complete')
        end = time.time()
        print('time for split', end - start)
        print(self.train[0].__dict__.keys())
        print(self.train[0].__dict__.values())

        custom_embeddings = vocab.Vectors(name='../SentEval/pretrained/glove.840B.300d.txt')

        # self.TEXT.build_vocab(self.train, self.dev, self.test, vectors = GloVe(name='840B', dim=300))
        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=custom_embeddings)
        self.LABEL.build_vocab(self.train)

        print('build vocab complete')
        self.LABEL.build_vocab(self.train)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = args.device

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       # TODO: change to args
                                       batch_size=64,
                                       device=device)
                                        # batch_size=args.batch_size,
                                        # device=args.gpu)

    def get_iterators(self):
        return self.train_iter, self.dev_iter, self.test_iter, self.TEXT, self.LABEL



