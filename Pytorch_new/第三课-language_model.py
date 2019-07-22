# @TIME : 2019/7/11 下午11:11
# @File : 第三课-language_model.py


import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

COMMON_PATH = '/Users/a1/Desktop/pytorch-7月/第二课资料/text8/'

USE_CUDA = torch.cuda.is_available()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000


# 定义torchtext
TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=COMMON_PATH, train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)

TEXT.build_vocab(train, max_size = MAX_VOCAB_SIZE)
print("vocab_size : {}".format(len(TEXT.vocab)))

VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits( (train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=32, repeat=False, shuffle=True)
it = iter(train_iter)   # ??  用法
batch = next(it)
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,1].data]))
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,1].data]))


# 定义模型
import torch
import torch.nn as nn

class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            # getattr(object, name[, default])
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_THNH':'tanh', 'RNN_RELU':'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for model was supplied, options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad), weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad)


model = RNNModel('LSTM', VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)
if USE_CUDA:
    model.moddel.cuda()
print('model ok')

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)  # 递归

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

def evaluate(model, data):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.

    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)

            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_loss += np.multiply(*data.size())
            total_loss += loss.item() * np.multiply(*data.size())

    loss = total_loss / total_count
    model.train()
    return loss


import copy
GRAD_CLIP = 1.
NUM_EPOCHS = 2

val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if i % 10 == 0:
            print("epoch", epoch, "iter", i, 'loss ', loss.item())

        # if i % 1000 == 0:
        #     val_loss = evaluate(model, val_iter)
        #     print('val_loss = ', val_loss)
        #
        #     if len(val_losses) == 0 or val_loss < min(val_losses):
        #         print("best model, val loss: ", val_loss)
        #         torch.save(model.state_dict(), "lm-best.th")
            # else:
            #     scheduler.step()
            #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # val_losses.append(val_loss)

