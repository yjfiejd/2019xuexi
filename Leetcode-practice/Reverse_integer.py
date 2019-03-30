# @TIME : 2019/3/28 下午11:14
# @File : Reverse_integer.py

#  自己写的方法：能用，但不够好，
# 1) 没有考虑越界
# 2）采用字符串方式操作，感觉有点啰嗦


import time


class stack:

    def __init__(self):
        self.item = []
    def isEmpty(self):
        return len(self.item) == 0

    def push(self, item):
        self.item.append(item)

    def pop(self):
        return self.item.pop()

    def peek(self):
        if not self.isEmpty():
            return self.item[len(self.item) - 1]
    def size(self):
        return len(self.item)


class reverse_list:

    def __init__(self, stack):
        self.stack = stack
        self.item = self.stack.item
        self.new_list = self.reverse_list()

    def reverse_list(self):
        new_list = []
        for i in range(len(self.item)):
            new_list.append(self.item.pop())
        return new_list


def reverse_int(num):

    num_str = str(num)

    if num > 0:
        int_stack = stack()
        temp = [int_stack.push(i) for i in num_str]
        int_reverse = reverse_list(int_stack)
        return int("".join(int_reverse.new_list))

    else:
        num_str = num_str.strip("-")
        int_stack = stack()
        temp = [int_stack.push(i) for i in num_str]
        int_reverse = reverse_list(int_stack)
        return int("-" + "".join(int_reverse.new_list))


start = time.time()

num = 12345
aa = reverse_int(num)
print(aa)   #-321

end = time.time()
print('第一个程序运行时间：', str(end - start ))




# 正确的写法： 利用 10进制，移动位置来操作，方便快捷！

# class solution:
#
#     def reverse(self, num):
#         """
#         :param x:  int
#         :return:  int
#         """
#         result = 0
#
#         abs_num = abs(num)
#
#         while(abs_num != 0):
#
#             temp = abs_num % 10
#             result = result * 10 + temp
#             abs_num = int(abs_num/10)  # 判断条件
#
#         if num > 0 and result < 2147483647:
#             return result
#         elif num < 0 and result <= 2147483647:
#             return -result
#         else:
#             return 0
#
#
#
# start2 = time.time()
#
# a = solution()
# print("\n", a.reverse(12345))
#
# end2 = time.time()
# print('第二个程序运行时间：', str(end2 - start2))