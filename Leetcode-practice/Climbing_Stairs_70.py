# @TIME : 2019/5/28 上午8:54
# @File : Climbing_Stairs_70.

# 可能性
# n = 4 (5种可能)
# 2 + 2
# 1 + 1 + 1 + 1
# 1 + 2 + 1
# 2 + 1 + 1
# 1 + 1 + 2

 # n = 5 (8种可能)
# 找规律？？？


# class Solution:
#     def climbStairs(self, n: int) -> int:
#         pre, cur = 0, 1
#         for i in range(n):
#             pre, cur = cur, pre+cur
#             return cur
#
#
import random

all = []
for i in range(800):
    all.append(i)


# for j in range(38):
#     nums = []
#     for i in range(8):
#         num = random.choices(all)
#         nums.append(num[0])
#         try:
#             num.remove(num)
#         except:
#             pass
#
#     print('num = ', nums)

n = 1
m = 1
for i in range(10000):
    random.shuffle(all)
    # print(all)
    num_index = all.index(158)

    if num_index < 600:
        print("次数 {} index = {}".format(n, num_index))
        m += 1

    else:
        print("未抽中次数 {}".format(n))
        n += 1

print("m = ", m)

