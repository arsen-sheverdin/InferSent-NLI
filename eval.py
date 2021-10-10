import argparse
import numpy as np
import sys
print(sys.executable)
import sklearn
sklearn.__version__
import nltk
nltk.download('punkt')
print('next')
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from pytorch_lightning import seed_everything
seed_everything(42)
from prepare_SNLI import *
from utils_eval import *



def evaluate(args):
    s = SNLI(args)

    train_iter, dev_iter, test_iter, text, label = s.get_iterators()

    cp_path = args.checkpoint_path

    if args.model_name == 'LSTM' or 'BiLSTM':
        # print('check')
        model_test = Recurrent.load_from_checkpoint(cp_path,
                                                    config=args,
                                                    text=text,
                                                  train_iter=train_iter,
                                                  dev_iter=dev_iter,
                                                  test_iter=test_iter)

        logger = TensorBoardLogger(args.path_for_logs, name= args.model_name)
        trainer = Trainer(
            logger=logger,
        )

    print(model_test.nli_net.encoder.embedding)
    trainer.test(model_test)


def main():
    parser = argparse.ArgumentParser()
    # Basic ARGs
    parser.add_argument('--model_name', type=str, default='LSTM',
                        help='The name of the model to experiment on: [not AWE!!!],  LSTM, BiLSTM, BiLSTM_pooling')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='Device in which we perform the operations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed')
    parser.add_argument('--checkpoint_path', type=str, default="lisalogs/LSTM/checkpoints/epoch=8-step=72963.ckpt",
                        help='Directory for LOADING(!!!) the models')
    parser.add_argument('--path_for_logs', type=str, default='Successful_eval',
                        help='Directory for saving(!!!) the models')
    # TODO: change to
    parser.add_argument('--tokenize', type=bool, default=True,
                        help='tokenization for SNLI class')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='Hidden representation`s size for recurrent models')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(torch.backends.cudnn.enabled)
    print(args)

    evaluate(args)


if __name__ == "__main__":
    main()