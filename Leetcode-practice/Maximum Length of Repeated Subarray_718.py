# @TIME : 2019/6/9 上午1:19
# @File : Maximum Length of Repeated Subarray_718.py

import numpy as np

# Given two integer arrays A and B, return the maximum length of an subarray that appears in both arrays.

# Input:
# A: [1,2,3,2,1]
# B: [3,2,1,4,7]
# Output: 3
# Explanation:
# The repeated subarray with maximum length is [3, 2, 1].

# Note
# 1 <= len(A), len(B) <= 1000
# 0 <= A[i], B[i] < 100

A = [1, 2, 3, 2, 1]
B = [3, 2, 1, 4]



def find(A, B):

    dp = [[0 for _ in range(len(A)+1)] for _ in range(len(B)+1)]
























# class Solution:
#     def __init__(self, A, B):
#         self.A = A
#         self.B = B
#
#     def findLength(self):
#         print('A = ', A)
#         print('B = ', B)
#
#         dp = [[0 for _ in range(len(self.B) + 1)] for _ in range(len(self.A) + 1)]
#         print(np.array(dp))
#         # 从第一格开始
#         for i in range(1, len(self.A) + 1):
#             print('i = ', i)
#             for j in range(1, len(self.B) + 1):
#                 print('j = ', j)
#                 print("value A[{}] {},  B[{}] {}".format(i-1, self.A[i-1], j-1, self.B[j-1]) )
#                 if self.A[i-1] == self.B[j-1]:
#                     dp[i][j] = dp[i-1][j-1] + 1
#                     print('\n')
#                     print(np.array(dp))
#                 else:
#                     dp[i][j] = 0
#
#         rel = max(max(row) for row in dp)
#         print('rel =', rel)
#         return rel
#
# a = Solution(A, B)
# a.findLength()
