# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
#
#
# m = nn.AvgPool1d(3, stride=2)
# c = m(torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]]))
# print('c = ', c)
#
# class Solution:
#     def find132pattern(self, nums):
#         third = float('-inf')
#         stack = []
#         for i in range(len(nums)-1, -1, -1):
#             if nums[i] < third:
#                 return False
#             else:
#                 while stack and stack[-1] < nums[i]:
#                     third = stack.pop()
#             stack.append(nums[i])
#         return True
#
# A = Solution()
# a = [1, 3, 2, 0]
# # a = [1, 2, 3, 4]
#
# cc = A.find132pattern(a)
# print(cc)