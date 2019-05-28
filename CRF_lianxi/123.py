# @TIME : 2019/4/16 上午9:10
# @File : 123.py

# @TIME : 2019/3/19 下午11:02
# @File : lstm_crf.py


# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    # 返回两个列表，一个是最大值列表，另一个是dim = 1,
    # 目的是取出当前的最大值在行向量中位置。
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
# 以前向算法的数值稳定方式计算log-sum-exp.
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


############################################################################## #
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 5
        self.hidden_dim = hidden_dim  # 3
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # 5
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)  # (25,5)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)  # (5, 1, num_layers =1, Bi=True)

        # Maps the output of the LSTM into tag space.将LSTM的输出映射到标记空间。
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)  # (3,5)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 转移参数矩阵. 条目i，j是*从* j转换*到* i 的分数。

        # 转移矩阵是训练的得来的，它是随机初始化的。
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 这两个语句强制执行约束: 我们从不转移到start-tag的，我们永远也不会从stop-tag转移.
        # 强制设置START和STOP的值为-10000
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 也就是第4行的位置全部设置为-10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # 也就是第5列的位置全部设置为-10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),  # (2,1,1)正态标准分布
                torch.randn(2, 1, self.hidden_dim // 2))  # (2,1,1)

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 使用前向算法来计算分区函数(部分函数)
        # torch.full:Returns a tensor of size :attr:`size` filled with :attr:`fill_value`.
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score. START_TAG拥有所有分数。
        # tensor([[-10000., -10000., -10000.,      0., -10000.]])
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 包装一个变量，以便我们获得自动反向提升(反向传播)
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep 在这个时间步长的前向张量
            for next_tag in range(self.tagset_size):
                # 发射得分。
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # 广播发射得分：不管以前的标记是什么都是相同的,发射得分来自BiLSTM训练的标签。
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # print(emit_score)

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                # trans_score的第i个entry的tag是从i转换到next_tag(也就是i+i的tag)的分数。#(1, 自动计算)-->(1, 5)
                trans_score = self.transitions[next_tag].view(1, -1)

                # 下一个标签的变量
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # next_tag_var的第i个条目是我们执行log-sum-exp之前的边（i - > next_tag）的值
                next_tag_var = forward_var + trans_score + emit_score

                # 所有得分。
                # The forward variable for this tag is log-sum-exp of all the scores.
                # 此标记的前向变量是所有分数的log-sum-exp的值。
                print(log_sum_exp(next_tag_var).view(1))
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)
            print(forward_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)

        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # 给出提供的标签序列的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]

        return score

    def _viterbi_decode(self, feats):
        """
        feats: LSTM 隐层到标签转变的向量。
        该函数使用于解码的，也就是说当训练时，是不需要使用的。
        """
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化viterbi向量在log空间
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        # tensor([[-10000., -10000., -10000., -10000., -10000.]])
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0  # 设置开始标签为0
        # tensor([[-10000., -10000., -10000.,      0., -10000.]])

        # forward_var at step i holds the viterbi variables for step i-1
        # 步骤i中的forward_var变量保持步骤i-1的viterbi变量
        forward_var = init_vvars  # tensor([[-10000., -10000., -10000.,      0., -10000.]])
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step# 保存这一步的后指针
            viterbivars_t = []  # holds the viterbi variables for this step# 保存这一步的viterbi向量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # next_tag_var [i]保存上一步标签i的维特比变量，加上从标签i转换到next_tag的分数。
                # 我们这里不包括发射分数，因为最大值不依赖于它们（我们在下面添加它们）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 现在添加发射分数，并将forward_var分配给我们刚刚计算的维特比变量
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # 转换到 STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        # 遵循后项指针来解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        # 弹出开始标签（我们不想把它归还给调用者）
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check #完整性检查
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        print("test---1")
        # 1、LSTM
        feats = self._get_lstm_features(sentence)
        # 2、前向算法
        forward_score = self._forward_alg(feats)
        # 3、句子得分
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # 从BiLSTM获得发射得分。
        # 一、lstm
        print("test---2")
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        # 找到最佳路径，给定的特征。
        # 二、进行viterbi解码
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


###############################################################################
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data 创建一些训练数据集
training_data = [("which genre of album is harder ... ..faster ?".split(), "O O O O O I I I O".split()),
                 ("the wall street journal reported today that apple corporation made money".split(),
                  "B I I I O O O B I O O".split()),
                 ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)  # 字典word_to_ix的length

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)  # (25,tag_to_ix, 5, 4 )
# 随机梯度进行优化
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training 训练之前检查预测的结果
# 使用 torch.no_grad()构建不需要追踪的上下文环境
with torch.no_grad():
    i = 0
    while i < 3:
        # training_data[0][0] token列表

        precheck_sent = prepare_sequence(training_data[i][0], word_to_ix)  # ￼￼￼[0,1,2,3,4,5,6,7,8,9,10]
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[i][1]], dtype=torch.long)
        # 返回标签所对应的长整型值
        print("before_train:", model(precheck_sent))
        i += 1
# ------------------------------------------------------------------------------

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        # 第一步，梯度清零
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # 第二步，转换为词为张量，转换标签为张量
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        # 第三步，前向传播
        loss = model.neg_log_likelihood(sentence_in, targets)
        model.train()

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        # 调用optimizer.step()来计算损失函数，梯度和更新参数
        loss.backward()
        optimizer.step()

# ------------------------------------------------------------------------------
# Check predictions after training。训练之后检查预测的结果
with torch.no_grad():
    i = 0
    while i < 3:
        precheck_sent = prepare_sequence(training_data[i][0], word_to_ix)

        # 此时会调用forward函数，返回 score,target_sequence 。
        print("after_train:", model(precheck_sent))
        i += 1
# We got it!