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


input_1 = "()[{}]"
input_2 = "{[]]}"

start1 = time.time()
aa = Solution1()
end1 = time.time()
time1 = end1 - start1
# print(aa.isValid(input_1))
print('第一个程序运行时间：', str(time1))

start2 = time.time()
bb = Solution2()
end2 = time.time()
time2 = end2 - start2
# print(bb.isValid(input_1))
print('第二个程序运行时间：', str(time2))

print("第一个程序时间减去第二个程序时间 = ", str(time1 - time2))
