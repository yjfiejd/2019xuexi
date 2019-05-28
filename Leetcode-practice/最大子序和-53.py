# @TIME : 2019/4/20 上午1:29
# @File : 最大子序和-53.py

# 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
#
# 示例:
#
# 输入: [-2,1,-3,4,-1,2,1,-5,4],
# 输出: 6
# 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
# 进阶:
#
# 如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。


import time

class Solution:
    def maxSubArray(self, nums) -> int:
        for i in range(1, len(nums)):
            nums[i] = nums[i] + max(nums[i-1], 0)
        return max(nums)



class Solution2:
    def maxSubArray(self, A):
        if not A:
            return 0
        curSum = maxSum = A[0]
        for num in A[1:]:
            curSum = max(num, curSum + num)
            maxSum = max(maxSum, curSum)

        return maxSum

t1 = time.time()
a = Solution()
b = a.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
t2 = time.time()
r1 = t2-t1
print('solution1 时间 = ', r1)

time.sleep(1)


t3 = time.time()
aa = Solution2()
bb = aa.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
t4 = time.time()
r2 = t4-t3
print('solution2 时间 = ', r2)

