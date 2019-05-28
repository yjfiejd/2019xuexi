import torch
import torch.nn as nn
import torch.optim as optm

torch.manual_seed(1)


def argmax(vec):
    # 返回vec size() 维度1（列） 上对应的每行的max值，与最大值的位置
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 前向算法的数值稳定方式计算log-sum-exp
def log_sum_exp(vec):  # vec.size()  [1, 5]
    max_score = vec[0, argmax(vec)]  # tensor(-0.9581)
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])    # tensor([[-0.9581, -0.9581, -0.9581, -0.9581, -0.9581]])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))




##################################################################################

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 想把词向量用多少的维度来表达
        self.hidden_dim = hidden_dim  # 隐藏层中的神经元个数 4
        self.vocab_size = vocab_size  # 25种
        self.tag_to_ix = tag_to_ix # 5种

        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # 将LSTM的输出映射到标记空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        print("self.hidden2tag = ", self.hidden2tag)

        # 转移参数矩阵, 就是BiLSTM中的参数矩阵(需要先随机初始化，训练过程会自动跟新)， entry i, j ——》是从j 转换到i的分数
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # 添加约束，不会有任何词语 tag -> start tag, 不会有 stop tag ->  任何词语tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 注意这是个矩阵
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()
        a = 1

    # 疑问
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装一个变量，以便反向传播
        forward_var = init_alphas
        print("forward_var = ", forward_var)

        # 举例：【我 爱 中华人民】计算total scores的时候，可以先算出【我， 爱】可能标注得所有情况，取 log_sum_exp
        # 然后再加上 最后一个词【中国人民】的 发射概率分数与转移概率分数
        # 其等价于【我，爱，中国人民】所有可能特征值指数次幂相加，然后取对数
        # 对句子迭代
        for feat in feats:
            print("\nfeat = ", feat)
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # 发射概率矩阵，注意需要变化形式，每个词对应多种tag概率，每个维度的长度都按最多种类进行扩展
                a = feat[next_tag]
                print("feat[next_tag = {}] = {}".format(next_tag, feat[next_tag])) # 第i个词的 发射概率 -> 拿出来做expand操作后 -> 才能与trans_score 相加

                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                print("emit_score = ", emit_score)

                # 转移概率矩阵
                trans_score = self.transitions[next_tag].view(1, -1)
                print("trans_score = ", trans_score)

                # 下一个标签的分数, 这里的forward_var 相当于文章中的 previous, 需要不断更新
                next_tag_var = forward_var + trans_score + emit_score
                print(" 中间forward_var = ", forward_var)
                print("next_tag_var = ", next_tag_var)

                # 前面的next_tag_var 算出来后 去 log-sum-exp 操作后 存入。下一次再用
                # print("\nlog_sum_exp(next_tag_var).view(1) = ", log_sum_exp(next_tag_var).view(1))
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                print("alphas_t = ", alphas_t)

            # 每个batch跑完，就是一行5个词语跑完后，这里共5个alphas_t分数，->cat汇总forward_var -> 最后一个状态的forward_var
            forward_var = torch.cat(alphas_t).view(1, -1)
            print("\nforward_var = ", forward_var)
        # 结尾分数
        print("self.transitions[self.tag_to_ix[STOP_TAG]] = ", self.transitions[self.tag_to_ix[STOP_TAG]])
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        print("terminal_var = ", terminal_var)
        alpha = log_sum_exp(terminal_var)
        print("alpha = ", alpha)

        return alpha


    def _get_lstm_features(self, sentence):
        # 应该是 获得 encoder 最后一层隐藏层参数
        self.hidden = self.init_hidden()
        # 这里的格式变换 注意下
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # 获得encoder 最后一层结果
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # 获得 feats
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 给出提供的标签序列的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            # 初始值 + 转移概率 + 发射概率
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            a = 1

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        a = 1


        return score


    def _viterbi_decode(self, feats):
        # 该还是用于解码，训练时，是不需要的
        backpointers = []

        # 初始化viterbi向量在log空间
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # 步骤i中的forward_var变量保持步骤i-1的viterbi变量
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  # 保存这时候的后指针, 有点像文章中的alpha0，最大值的index
            viterbivars_t = []  # 保存这步的viterbi向量
            for next_tag in range(self.tagset_size):
                # next_tag_var[i]保存上一步标签i的维特比变量，加上从标签i转换到next_tag的分数, 我们这里不包括发射分数，因为最大值不依赖于它们（我们在下面添加它们）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # 现在添加发射分数，并将forward_var分配给我们刚刚计算的维特比变量
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)


        # 转换到 STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 遵循后项指针来解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)


        # 弹出开始标签
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood(self, sentence, tags):
        print("test1 ...")
        # LSTM
        feats = self._get_lstm_features(sentence)
        a = 1
        # forward算法
        forward_score = self._forward_alg(feats)
        print("结果 forward_score = ", forward_score)
        # 句子分数
        gold_score = self._score_sentence(feats, tags)
        print("gold_score = ", gold_score)
        return forward_score - gold_score



    def forward(self, sentense):
        # 1,lstm
        print("test2 ...")
        lstm_feats = self._get_lstm_features(sentense)

        # 2, viterbi 解码
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


##################################################################################


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# 训练集
training_data = [("which genre of album is harder ... ..faster ?".split(), "O O O O O I I I O".split()),
                 ("the wall street journal reported today that apple corporation made money".split(),
                  "B I I I O O O B I O O".split()),
                 ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]

a = 1
word_to_ix = {}
for sentence, tag in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B":0, "I":1, "O":2, START_TAG:3, STOP_TAG:4}

# (25, 5, 5, 4)
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optm.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


# 训练之前检查预测的结果
with torch.no_grad():
    i = 0
    a = 1
    while i < 3:
        print("\ntraining_data[i][0] = ", training_data[i][0])
        precheck_sent = prepare_sequence(training_data[i][0], word_to_ix)  # ￼￼￼[0,1,2,3,4,5,6,7,8,9,10]

        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[i][1]], dtype=torch.long)
        print("precheck_tags = ", precheck_tags)
        a = 1
        print("before_train: ", model(precheck_sent))
        i +=1

# ------------------------------------------------------------------------------
for epoch in range(1):
    print('----------------------------------')

    print('\nepoch: ', epoch)

    for sentence, tags in training_data:

        # 1，梯度清零
        model.zero_grad()

        # 2, 转换词向量
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        print("sentence_in = ", sentence_in)
        print("targets = ", targets)



        # 3, 前向传播
        loss = model.neg_log_likelihood(sentence_in, targets)
        model.train()

        # 4, 更新梯度参数，反向传播
        loss.backward()
        optimizer.step()


# 训练之后检查预测的结果

print(" \n 测试开始。。。。。")
with torch.no_grad():
    i = 0
    while i < 3:
        precheck_sent = prepare_sequence(training_data[i][0], word_to_ix)

        # 此时会调用forward函数，返回 score,target_sequence 。
        print("after_train:", model(precheck_sent))
        i += 1




