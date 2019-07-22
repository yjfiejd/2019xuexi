# @TIME : 2019/7/9 下午8:42
# @File : FizzBuzz_new.py


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
from torch.autograd import Variable

# 准备数据
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0
# training data -> 注意这里没有扯什么反转
NUM_DIGITS = 10  # 十个位数来表达

# 注意后面是923个数据，选择不同的batch，只会影响到最后一波剩下的个数，然后计算准确率，最后每次都会更新一次epoch的准确率
# dataloader 中已经预设了 batch，跑完一波，就计算一下
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2**NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i) for i in range(101, 2**NUM_DIGITS)])
print("trX[:5] = \n",trX[:5])
print('--------------')
print("trY[:5] = \n",trY[:5])


# 准备模型
class FizzBuzzModel(nn.Module):
    def __init__(self, in_features, out_classes, hidden_size, n_hidden_layers):
        super(FizzBuzzModel, self).__init__()

        # 注意有多层的hidden layers
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())

        self.inputLayer = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()

        self.layers = nn.Sequential(*layers)

        self.outputLayer = nn.Linear(hidden_size, out_classes)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.relu(x)
        x = self.layers(x)
        out = self.outputLayer(x)
        return out


# 训练模型
# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FizzBuzzModel(10, 4, 150, 3).to(device)
print("model = ", model)
learning_rate = 0.02
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader  $ 有点迷
FizzBuzzDataset = tud.TensorDataset(torch.from_numpy(trX).float().to(device),
                                    torch.from_numpy(trY).long().to(device))
dataloader = tud.DataLoader(dataset=FizzBuzzDataset, batch_size=300, shuffle=True)

# 训练
model.train()
for epoch in range(1, 300):
    for i, (batch_x, batch_y) in enumerate(dataloader):
        # print('batch_x= ', batch_x, 'shape = ', batch_x.shape)
        # print('batch_y= ', batch_y, 'shape = ', batch_y.shape)

        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    # print('out.data', out.data, out.data.shape)
    _, predicted = torch.max(out.data, 1)
    # print('predicted = ', predicted, predicted.shape)
    total += batch_y.size(0)
    correct += (predicted == batch_y).sum().item()
    acc = 100*correct/total
    print('Epoch : {:0>4d} | Loss : {:<6.4f} | Train Accuracy : {:<6.2f}%'.format(epoch,loss,acc))  #小数点后面的位数


# 模型测试
def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


# 模型预测：需要切换为 eval模式
model.eval()

testX = np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
predicts = model(torch.from_numpy(testX).float().to(device))

_, res = torch.max(predicts, 1)

predictions = [fizz_buzz_decode(i, prediction) for (i, prediction) in zip(range(1, 101), res)]

print('------------------------------')
print('prediction = \n', predictions )





























