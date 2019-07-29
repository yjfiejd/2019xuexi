# @TIME : 2019/7/23 上午23:57
# @File : 第四课.py

import torch
import random
import time
import spacy
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""
代码逻辑：

1）用torchtext中自带的datasets下载 IMDB -> 获得 train_data, test_data
    1.1 需要把train_data 切分为 train_data, valid_data (默认是7：3)
2) 利用torchtext中的data 构造vocab等后面需要用的一些字典. build vocab -> TEXT & LABEL
3) 把单条的训练数据 整合为batch 利用：BucketIterator（ batches examples of similar lengths together） -> train_iterator, valid_iterator, test_iterator
4）创建 Averaging model, 主要对embedding后的结果做pooling 再送入Linear layer，需要定义模型的参数： vocab_size, embeding_size, output_size, pad_idx（注意这个别漏了）
5）model 建立后，第一需要载入预训练的词向量，vector，来自TEXT build vocab时候已导入glove, 第二需要把 <unk> <pad> 初始化weight中置为0
6）训练model，loss采用 BCEWithLogitsLoss -> 就是把Sigmoid-BCELoss合成在一起（This loss combines a Sigmoid layer and the BCELoss in one single class.）
    6.1 https://blog.csdn.net/qq_22210253/article/details/85222093
    6.2 https://pytorch.org/docs/stable/nn.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss
7）准确率acc：binary_accuracy，用torch.round四舍五入后再用.float()转为0， 1 方便统计   
8）记录validation loss 使用model.state_dict()保存，如果后续需要调用则，model.load_state_dict(torch.load('xxx.pt'))

"""



# 1）下载数据，train & test
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

print("\ndowning IMDB data...")
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print('finished...')


print('---------------')
print(vars(train_data.examples[0]))
print('---------------')

# 2) 切分数据 train valid
print("\n切分数据 train valid data...")
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print("Number of training examples: ", len(train_data))
print("Number of validation examples: ", len(valid_data))
print("Number of testing examples: ", len(test_data))
print('finished...')



# 3) build vocab， 并定义max_size  & glove
print("\nbuild vocab， 并定义max_size  & glove...")
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init = torch.Tensor.normal_)
LABEL.build_vocab(test_data)

print('---------------')
print("TEXT.vocab.freqs.most_common(20)",TEXT.vocab.freqs.most_common(20))
print('---------------')
print("TEXT.vocab.itos[:10]",TEXT.vocab.itos[:10])
print('---------------')
print('finished')


# 4) 创建 iterator batch-examples
print("\n创建 iterator batch-examples...")
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)
print('finished')


## 5) Word Averaging 模型
class WordAvg(nn.Module):
    def __init__(self, vocab_size, embeding_size, output_size, pad_idx):
        super(WordAvg, self).__init__()
        self.embed = nn.Embedding(vocab_size, embeding_size, padding_idx=pad_idx)
        self.fc = nn.Linear(embeding_size, output_size)

    def forward(self, text):
        embed = self.embed(text)  # sen_length, batch_size, embed_size
        embed = embed.permute(1, 0, 2)  # batch_size, sen_length, embed_size
        pooled = F.avg_pool2d(embed, (embed.shape[1], 1)).squeeze(1)  # batch_size, embed_size
        return self.fc(pooled)


# 6) 定义模型参数
print("\n定义模型参数 & 创建模型...")
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1  # (积极，消极)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # string to index, pad的序号是啥
model = WordAvg(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
print('finished')


# 7) 载入预训练的词向量 初始化 UNK, PAD 矩阵中的值为0
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# 8）训练模型
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()  # 多分类的cross entropy
model = model.to(device)
criterion = criterion.to(device)

# 计算准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))  #把概率的结果 四舍五入
    correct = (rounded_preds == y).float()  # True False -> 转为 1， 0
    acc = correct.sum() / len(correct)
    return acc

# train
def train(model, iterator, optimizer, criterion):
    """
    :param model:  传入模型
    :param iterator: 传入多个 batch 组成的输入
    :param optimizer: 传入优化算法 optimizer
    :param criterion: 传入计算loss的方法
    :return:
    """
    epoch_loss = 0
    epoch_acc  = 0
    model.train()  # 注意切换模式

    for batch in iterator:  # 有多少个batch
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)  # 在第1个维度上 去除维度=1
        loss = criterion(predictions, batch.label)  # 计算loss，用于backward()
        acc = binary_accuracy(predictions, batch.label)  # 计算acc，看下这次batch的准确度
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 取出loss中的值
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.eval()

        with torch.no_grad():
            for batch in iterator:
                prediction = model(batch.text).squeeze(1)
                loss = criterion(prediction, batch.label)
                acc = binary_accuracy(prediction, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss/len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(end_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

print("\n训练开始...")
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')
        print('save ok, valid_loss {}'.format(valid_loss))


    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


a = 1
print('ok')
print("\n预测开始...")

nlp = spacy.load('en')  # 这就是个tokenizer
def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    print('tokenized :', tokenized)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    print('indexed :', indexed)
    tensor = torch.LongTensor(indexed).to(device)
    print('tensor :', tensor)
    tensor = tensor.unsqueeze(1)
    print('tensor after unsqueeze(1) :', tensor)
    prediction = torch.sigmoid(model(tensor))
    print("prediction.item() :", prediction)
    return prediction.item()

sen = "This film is terrible"
print('\n预测 sen = ', sen)
print('预测 结果:', predict_sentiment(sen))

sen = "This film is great"
print('\n预测 sen = ', sen)
print('预测 结果:', predict_sentiment(sen))

sen = "The film is very good！"
print('\n预测 sen = ', sen)
print('预测 结果:', predict_sentiment(sen))








# RNN 模型  (这里使用最后一个 LSTM hidden state hT 来表示整个句子)

# class RNN(nn.Module):
#
#     def __init__(self, vocab_size, embedding_size, hidden_dim, output_dim,
#                  n_layers, bidirectional, dropout, pad_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
#         self.run = nn.LSTM(embedding_size, hidden_dim, num_layers=n_layers,
#                             bidirectional=bidirectional, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text):
#         embedded = self.dropout(self.embedding(text))
#         ouput, (hidden, cell) = self.rnn(embedded)    # 这里面的维度，每一种维度需要搞清楚
#         # output: [sen_len, batch_size, hid_dim * num directions]
#         # hidden = [num_layers * num direction, batch_size, hid_dim]
#         # cell = [num_layers * num direction, batch_size, hid_dim]
#
#         # concat the final forward(hidden[-2, :, :]) and backward (hidden[-1, :, :]) hidden layers
#         hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         return self.fc(hidden.squeeze(0))


















