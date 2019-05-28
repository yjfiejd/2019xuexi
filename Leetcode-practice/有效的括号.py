# @TIME : 2019/4/2 上午8:36
# @File : 有效的括号.py

import time


class Solution1:

    def isValid(self, s):

        stack = []
        dict1 = {"(": ")", "{": "}", "[": "]"}

        for elem in s:
            if elem in dict1.keys():
                stack.append(elem)
            elif elem in dict1.values():
                if stack == [] or dict1[stack.pop()] != elem:
                    return False
            else:
                return False
        return stack == []


class Solution2:
    def isValid(self, s):
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''


# --------------------

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

class Solution3:

    def isValid(self, s):

        lefty = "({["
        righty = ")}]"
        S = ArrayStack()
        for c in s:
            if c in lefty:
                S.push(c)
            elif c in righty:
                if S.is_empty():
                    return False
                if righty.index(c) != lefty.index(S.pop()):
                    return False
        return S.is_empty()


input_1 = "()[{}]"
input_2 = "{[]]}"

start1 = time.time()
aa = Solution1()
end1 = time.time()
time1 = end1 - start1
# print(aa.isValid(input_1))
print('第一个程序运行时间：', str(time1))

time.sleep(1)

start2 = time.time()
bb = Solution2()
end2 = time.time()
time2 = end2 - start2
# print(bb.isValid(input_1))
print('第二个程序运行时间：', str(time2))

time.sleep(1)

start3 = time.time()
bb3 = Solution3()
end3 = time.time()
time3 = end3 - start3
# print(bb.isValid(input_1))
print('第三个程序运行时间：', str(time3))


print("三个程序，最短时间 = ", min(time1, time2, time3))
