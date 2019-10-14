# @TIME : 2019/9/13 下午8:38
# @File : 颜色分类_排序_75.py

"""
给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

注意:
不能使用代码库中的排序函数来解决这道题。
示例:

输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
进阶：

一个直观的解决方案是使用计数排序的两趟扫描算法。
首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
你能想出一个仅使用常数空间的一趟扫描算法吗？

"""


class Solution:

    # def sortColors(self, nums):
    #     """
    #     不能新建空间，那就用for 重新赋值，count 种类个数 = for 的 index
    #     """
    #     num_0 = nums.count(0)
    #     num_1 = nums.count(1)
    #     for i in range(0, num_0):
    #         nums[i] = 0
    #     for i in range(num_0, num_0 + num_1):
    #         nums[i] = 1
    #     for i in range(num_0 + num_1, len(nums)):
    #         nums[i] = 2
    #     print(nums)

    def sortColors(self, nums):
        """
        原来这是荷兰国旗问题，失敬失敬
        通过3个指针进行操作，也是for 循环一次, 时间空间复杂度同上
        """
        p1 = cur = 0
        p2 = len(nums)-1

        while cur <= p2:
            if nums[cur] == 0:
                nums[cur],  nums[p1] = nums[p1], nums[cur]
                cur += 1
                p1 += 1
            elif nums[cur] == 2:
                nums[cur], nums[p2] = nums[p2], nums[cur]
                p2 -= 1
            else:
                cur += 1

        print(nums)






if __name__ == "__main__":
    nums = [2,0,2,1,1,0]
    A = Solution()
    A.sortColors(nums)












