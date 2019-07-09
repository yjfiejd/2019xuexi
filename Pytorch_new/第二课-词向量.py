# @TIME : 2019/7/4 下午11:04
# @File : 第二课-词向量.py

# 【Dataloader】
# 需要把所有的text编码成数字，然后subsampling预处理这些文字
# 保存vocabulary，单词count, normalized word frequency
# 每一个 iteration sample 一个中心词
# 根据当前的中心词返回 context 单词
# 根据中心词sample 一些negative单词
# 返回单词的counts


# 导入常用的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter
from collections import Counter
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_distances


# 设置random seed，保证每次结果一样
USE_CUDA = torch.cuda.is_available()
print('USE_CUDA = ', USE_CUDA)
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

# 设置超参数
K = 100   # negative sampling 中取样，每次取一个正确sample，需要取K个错误的sample
C = 3     # 每个单词周围的窗口
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 1280
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100
LOG_FILE = "word-embedding.log"

# 【数据的预处理】
print("正在处理: 读取数据, 获得text.split...")
def word_tokenize(text):
    """本次数据都是英文，都用空格分割了，且无其他符合干扰，所以用split()"""
    return text.split()
train_path = "/Users/a1/Desktop/pytorch-7月/第二课资料/text8/text8.train.txt"
with open(train_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("正在处理: 构造所有词, Counter->dict {word:counts}, <unk>->dict,  ...")
# 切词：从文本中读取所有的文字，通过这些文本创建一个vocabulary
text = [w for w in word_tokenize(text.lower())]
# 因为文本单词数量太大，我们只选取我常见的 MAX_VOCAB_SIZE个单词（包含unk，不常见单词）# 获得：MAX_VOCAB_SIZE-1个词的次数  获得：unk的次数
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))

print("正在处理: word_to_idx {word:i}...")
# word_to_idx
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word:i for i, word in enumerate(idx_to_word)}


print("正在处理: word_freqs -> counts/sum_counts...")
# word_freqs
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts/np.sum(word_counts)
VOCAB_SIZE = len(idx_to_word)

a = 1


class WordEmbeddingDataset(tud.Dataset):

    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        """
        - 中心词
        - 单词附近的 positive 单词
        - 随机采样的K个单词作为 negative sample
        """
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]  # Tensor 可以这样操作, 送入list
        neg_words = torch.multinomial(self.word_freqs, K*pos_words.shape[0], True)
        return center_word, pos_words, neg_words

# 创建dataset 和 dataloader
print("正在处理: 创建dataset & dataloader...")
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


print("正在处理: 定义Pytorch模型 -> init vocab_size & embed_size -> forward_func ...")

#
class EmbeddingModel(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size  #?? 为啥，为了后续数学求导方便？

        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)  # ??? 为什么

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_labels, pos_labels, neg_labels):
        """主要是 embedding -> 获得loss
        # input_label: [batch_size] 中心词的idx, 只有一维
        # pos_labels: [batch_size, (window_size * 2)] 中心词周围 window 出现过的单词
        # neg_labels: [batch_size, (window_size * 2 * K)] 中心词周围没有出现过的单词，从negative sampling 得到

        # 在 input_embedding 维度为2的上面 :unsqueeze -> 加上1个维度 -> 【batch_size, embedding_size，1】
        # If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
        # log_pos  ->  维度是[batch_size, (windows_size*2), 1]
        # squeeze(): 去除size为1的维度，包括行和列。至于维度大于等于2时，squeeze()不起作用。
        # log_pos  -> [batch_size, (windows_size*2)】
        """

        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # 【batch_size, embedding_size】-》tensor
        pos_embedding = self.out_embed(pos_labels)  # 【batch_size, (windows_size*2), embedding_size】
        neg_embedding = self.out_embed(neg_labels)  #  [batch_size, (windows_size*2*k), embedding_size]

        # 计算 skip-gram loss
        input_embedding = input_embedding.unsqueeze(2)
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze(2)

        log_pos = F.logsigmoid(pos_dot).sum(1)
        neg_pos = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + neg_pos

        return -loss  # [batch_size]


    def input_embedding(self):
        """ 取出模型 """
        return self.in_embed.weight.data.numpy()


print("正在处理：定义评估模型的代码...")

def evaluate(filename, embedding_weights):
    if filename.endwith('.csv'):
        data = pd.read_csv(filename, sep=',')
    else:
        data = pd.read_csv(filename, sep='\t')

    human_similarity = []
    model_similarity = []

    for i in data.ilod[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))

    return scipy.stats.spearmanr(human_similarity, model_similarity)  # model_similarity

def find_nearest(word, embedding_weights):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]


print("正在处理: 训练Pytorch模型...")
"""
# 训练模型
# - 模型需要若干个epoch
# - 每个epoch 我们都把所有的数据分成若干个 batch
# - 把每个batch的输入和输出都包装成cuda tensor
# - forward pass 通过输入的句子预测每个单词的下一个单词
# - 计算 cross entropy loss
# - backward pass
# - 更新模型参数
# - 每隔一定的iteration输出模型在当前iteration的loss

"""

model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            with open(LOG_FILE, "a") as f:
                f.write("epoch:{}, iter:{}, loss:{} \n".format(e, i, loss.item()))


    embedding_weights = model.input_embedding()
    np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))

















































































