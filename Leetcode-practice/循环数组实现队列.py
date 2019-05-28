# @TIME : 2019/4/7 下午12:51
# @File : 循环数组实现队列.py


# 最简单的方法不好，pop(0), 每次弹出时间复杂度总是O(N), 避免这种方法，最好做到O(1)
# 使用循环数组的方式，前面用空指针代替，循环第一个元素取模的方式做

class Empty(Exception):
    pass

class ArrayQueue:

    DEFALUT_CAPACITY = 10

    def __init__(self):
        self._data = [None] * ArrayQueue.DEFALUT_CAPACITY
        self._size = 0
        self._front = 0

    def __len__(self):
        return self._size

    def is_empyty(self):
        return self._size == 0

    def first(self):
        if self.is_empyty():
            raise Empty("The queue is empty")
        return self._data[self._front]

    def dequeue(self):
        if self.is_empyty():
            raise Empty("The queue is empty")
        # 保存取出的元素
        answer = self._data[self._front]

        # 列表中原始front置为空，修改front位置，size-1
        self._data[self._front] = None
        self._front = (self._size + 1) % len(self._data)
        self._size -= 1

        return answer

    def enqueue(self, e):

        if self._size == self._data:
            self._resize(2 * len(self._data))
        # 插入元素位置，front + size，现有元素位置
        avail = (self._front + self._size) % len(self._data)
        self._data[avail] = e
        self._size += 1

    def _resize(self, cap):
        # 扩展原始数组
        old = self._data
        self._data = [None] * cap
        walk = self._front

        # 把原始小数组中的元素拷贝至新数组，且新数组front从0开始
        for i in range(len(self._data)):
            self._data[i] = old[walk]
            walk = (walk + 1) % len(old)

        self._front = 0





a = ArrayQueue()
a.enqueue(5)
a.enqueue(10)
a.enqueue(11)
a.enqueue(12)


print("queue = " ,a._data)
b = a.dequeue()
print("queue = ", a._data)

print("b = ", b)


aa = 1