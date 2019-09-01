# @TIME : 2019/8/5 上午9:25
# @File : 三数之和_15.py

import time

"""
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

"""
#
# # 方法一：利用自带库操作 且去重，最好别用自带库
# from itertools import combinations
# def three_sum0(nums):
#     """自己写的，可以去重"""
#
#     all = [i for i in combinations(nums, 3) if sum(i)== 0]
#
#     new = {}
#     for i in all:
#         b = set()
#         for j in i:
#             b.add(j)
#         new.update({i:b})
#
#     stack1 = []
#     answ = []
#     for k, v in new.items():
#         if not stack1:
#             stack1.append(v)
#             answ.append(list(k))
#         else:
#             if v in stack1:
#                 pass
#             else:
#                 stack1.append(v)
#                 answ.append(list(k))
#
#     return all
#
# start = time.time()
# nums = [-1, 0, 1, 2, -1, -4]
# print(three_sum0(nums))
# end = time.time()
# print(end-start)
# #
# #
# #
# # 方法二： 三成循环: O(n^3)
# def three_sum1(nums):
#     """这种暴力方法，n的3次方的方法，有重复情况"""
#     all = []
#     for i in range(len(nums) - 2):
#         for j in range(i+1, len(nums)-1):
#             for k in range(j+1, len(nums)):
#                 if nums[i] + nums[j] + nums[k] == 0:
#                     all.append([nums[i], nums[j], nums[k]])
#     return all
# start = time.time()
# nums = [-1, 0, 1, 2, -1, -4]
# print(three_sum1(nums))
# end = time.time()
# t2 = end - start
# print(end-start)


# 方法三： 双指针操作: O(n^2)
def three_Sum3(nums):

    res = []
    nums.sort()
    length = len(nums)

    for i in range(length-2):
        if nums[i] > 0: break
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i + 1,length - 1
        while l < r:
            sums = nums[i] + nums[l] + nums[r]

            if sums < 0:
                l += 1
            elif sums > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l< r and nums[l] == nums[l+1]:  # 注意这里也有限制条件
                    l += 1
                while l< r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1
                r -= 1
    return res


start1 = time.time()
# nums = [ 0, -1, 1, 2, -1, -4]
nums = [ 0, 0, 0]
print(three_Sum3(nums))
end1 = time.time()
t3 = end1 - start1
print(end1-start1)


# print()
# print(t2 - t3)





