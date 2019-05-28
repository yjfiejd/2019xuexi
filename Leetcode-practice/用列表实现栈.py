# @TIME : 2019/4/7 上午9:12
# @File : 用列表实现栈.py


class Empty(Exception):
    pass

class ArrayStack:

    def __init__(self):
        self._data = []

    def len(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def push(self, item):
        return self._data.append(item)

    def top(self):
        if self.is_empty():
            raise Empty("Stack is empty")
        return self._data[-1]

    def pop(self):
        if self.is_empty():
            raise Empty("Stack is empty")
        return self._data.pop()







