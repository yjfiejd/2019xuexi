# @TIME : 2019/7/1 上午9:00
# @File : numpy_实现两层神经网络.py
import numpy as np

# 全连接 Relu 神经网络，一个隐藏层，没有bias -> 用来从x预测y，使用L2 Loss
# h = w1*X + b1
# a = Max(0, h)
# y_hat = w2*a + b2


# 假设64个训练数据，输入是1000维度，隐藏层是100维度，输出层是10维
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)  # 把1000维转换为100维
w2 = np.random.randn(H, D_out)

# learning rate
learning_rate = 1e-6

for it in range(500):
    # forward pass
    h = x.dot(w1)  # N * H
    h_relu = np.maximum(h, 0)  # N * H
    y_pred = h_relu.dot(w2)  # N * D_out

    # compute loss
    loss = np.square(y_pred - y).sum()
    print(it, loss)

    # backward pass
    # compute gradient
    grad_y_pred = 2.0 * (y_pred - y)  # 因为刚才是loss是平方，求导后2放前面了
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    # update w1, and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2



