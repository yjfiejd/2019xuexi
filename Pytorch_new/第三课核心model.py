# @TIME : 2019/7/23 上午8:27
# @File : 第三课核心model.py

import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 5000
COMMON_PATH = '/Users/a1/Desktop/pytorch-7月/第二课资料/text8/'


TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=COMMON_PATH,
    train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))

VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=32, repeat=False, shuffle=True)



class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            pass

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
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, requires_grad = True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad)




model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)


def evaluate(model, data):
    model.eval()
    total_loss = 0
    it = iter(data)
    total_count = 0
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item() * np.multiply(*data.size())

    loss = total_loss / total_count
    model.train()
    return loss

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)



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
            print("epoch", epoch, "iter", i, "loss", loss.item())

        if i % 20 == 0:
            val_losses = evaluate(model, val_iter)
            if len(val_losses) == 0 or val_losses < min(val_losses):
                print('best model, val loss:', val_losses)
                torch.save(model.state_dict(), 'lm-best.th')
            else:
                scheduler.step()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            val_losses.append(val_losses)


# epoch 0 iter 0 loss 8.51561164855957
# epoch 0 iter 10 loss 6.387548923492432
# epoch 0 iter 20 loss 6.047845363616943
# epoch 0 iter 30 loss 5.752002716064453
# epoch 0 iter 40 loss 5.883333683013916
# epoch 0 iter 50 loss 5.7453083992004395
# epoch 0 iter 60 loss 5.822821140289307
# epoch 0 iter 70 loss 5.691372871398926
# epoch 0 iter 80 loss 5.84808349609375
# epoch 0 iter 90 loss 5.586045265197754


best_model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)
best_model.load_state_dict(torch.load('lm-best.th'))

val_loss = evaluate(best_model, val_iter)
print("perplexity: ", np.exp(val_loss))

















