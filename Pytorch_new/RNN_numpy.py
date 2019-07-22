# @TIME : 2019/7/21 下午02:07
# @File : RNN_numpy.py


import numpy as np
from functools import reduce

class ReluActivator:
    """relu 激活函数"""
    def forward(self, weighted_input):
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0

class IdentityActivator:
    """线性的激活函数"""
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1

class SigmoidActivator:
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

class TanhActivator:
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2*weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output


def element_wise_op(array, op):
    """
    https://numpy.org/devdocs/reference/arrays.nditer.html
    Modifying Array Values， 可以对每个元素自定义func操作
    """
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


class RecurrentLayer:
    def __init__(self, input_width, state_width, activator, learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate

        self.times = 0  # 初始化 t0 时刻
        self.state_list = []  # 保存各个时刻state
        self.state_list.append(np.zeros((state_width, 1))) # 初始化s0
        self.U = np.random.uniform(-1e-4, 1e-4, (state_width, input_width))  # U: 外部输入n维度, state为m维度， n*m
        self.W = np.random.uniform(-1e-1, 1e-4, (state_width, state_width))  # W: 外部输入, n*n

    def forward(self, input_array):
        """st = f(Uxt + Wst-1)"""
        self.times += 1
        state = (np.dot(self.U, input_array) + np.dot(self.W, self.state_list[-1]))  # 注意使用上一次state的值
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)

    def backward(self, sensitivity_array, activator):
        """BPTT"""
        self.calc_delta(sensitivity_array, activator)
        self.calc_gradient()

    def update(self):
        """更新 w"""
        self.W -= self.learning_rate * self.gradient

    def calc_delta(self, sensitivity_array, activator):
        self.delta_list = []  # 用于保存各个时刻误差项
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width, 1)))

        self.delta_list.append(sensitivity_array)
        for k in range(self.times - 1, 0, -1):
            self.calc_delta_k(k, activator)

    def calc_delta_k(self, k, activator):
        """根据 k+1 时刻的delta计算 k 时刻的delta"""
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1], activator.backward)
        self.delta_list[k] = np.dot(np.dot(self.delta_list[k+1].T, self.W), np.diag(state[:, 0])).T  # 误差项计算，这里是连乘

    def calc_gradient(self):
        self.gradient_list = []  # 保存每个时刻的权重梯度
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros((self.state_width, self.state_width)))

        for t in range(self.times, 0, -1):
            self.calc_gradient_t(t)
        # 实际梯度是各个时刻的梯度之和
        self.gradient = reduce(lambda  a, b: a+b, self.gradient_list, self.gradient_list[0])

    def calc_gradient_t(self, t):
        """计算每个时刻t的权重的梯度 W """
        gradient = np.dot(self.delta_list[t], self.state_list[t-1].T)  # 注意知道任意时刻t的误差 & t-1时刻的St-1输出，可以得到t时刻的权重gradient
        self.gradient_list[t] = gradient

    def reset_state(self):
        """初始化 t0, s0"""
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((self.state_width, 1)))


def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d



def gradient_check():
    """梯度检查"""
    error_function = lambda x: x.sum()

    rl = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

    x, d = data_set()
    print(x)
    print(d)

    rl.forward(x[0])
    rl.forward(x[1])

    sensitivity_array = np.ones(rl.state_list[-1].shape, dtype=np.float64)
    rl.backward(sensitivity_array, IdentityActivator())
    # rl.backward(sensitivity_array,  ReluActivator())

    epsilon = 10e-4
    for i in range(rl.W.shape[0]):
        for j in range(rl.W.shape[1]):
            rl.W[i, j] += epsilon
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err1 = error_function(rl.state_list[-1])

            rl.W[i, j] -= 2*epsilon  # 梯度下降
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err2 = error_function(rl.state_list[-1])

            expect_grad = (err1 - err2) / (2 * epsilon)
            rl.W[i, j] += epsilon
            print('weights(%d,%d): expected - actural %f - %f' % (i, j, expect_grad, rl.gradient[i, j]))



def run():
    l = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
    x, d = data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(d, ReluActivator())
    return l


# run()
# gradient_check()
















