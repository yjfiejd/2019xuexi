# @TIME : 2019/8/6 上午7:32
# @File : 最小栈155.py

import time


# 方法一：使用两个栈，存储数据，和存储最小元素
class MinStack1:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.helper = []

    def push(self, x: int) -> None:
        self.data.append(x)
        if not self.helper or x < self.helper[-1]:
            self.helper.append(x)
        else:
            self.helper.append(self.helper[-1])

    def pop(self) -> None:
        if self.data:
            self.helper.pop()
            return self.data.pop()

    def top(self) -> int:
        if self.data:
            return self.data[-1]

    def getMin(self) -> int:
        if self.helper:
            return self.helper[-1]


# 方法二：使用一个栈，但每次存入的元素是list
class MinStack2:

    def __init__(self):
        self.data = [[-1, float('inf')]]

    def push(self, x):
        self.data.append([x, min(x, self.data[-1][-1])])

    def pop(self):
        if len(self.data) > 1:
            return self.data.pop()
    def top(self):
        if len(self.data) == 1: return None
        return self.data[-1][0]

    def getMin(self):
        return self.data[-1][1]


if __name__ == "__main__":

    s1 = time.time()
    minStack = MinStack1()
    print(minStack.push(-2))
    print(minStack.push(0))
    print(minStack.push(-3))
    print(minStack.getMin())
    print(minStack.pop())
    print(minStack.top())
    print(minStack.getMin())
    e1 = time.time()
    print("方法一",e1-s1)

    # time.sleep(1)
    print("-"*20)
    s2 = time.time()
    minStack = MinStack2()
    print(minStack.push(-2))
    print(minStack.push(0))
    print(minStack.push(-3))
    print(minStack.getMin())
    print(minStack.pop())
    print(minStack.top())
    print(minStack.getMin())
    e2 = time.time()
    print("方法二",e2-s2)