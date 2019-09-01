# @TIME : 2019/9/1 下午5:00
# @File : Merge_sorted_array_88.py



"""
[easy]
给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。

说明:

初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
示例:

输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]

"""
import time


class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        不用去开辟额外空间
        """
        nums1[m:] = nums2[:n]
        nums1.sort()
        return nums1

    def merge2(self, nums1, m, nums2, n):
        """双指针法"""
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1
        while p1 > 0 and p2 > 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -= 1
        nums1[:p2 + 1] = nums2[:p2 + 1]
        return nums1



if __name__ == "__main__":

    nums1 = [1,2,3,0,0,0]
    m = 3
    nums2 = [2,5,6]
    n = 3


    A = Solution()

    t1 = time.time()
    print(A.merge(nums1, m, nums2, n))
    t2 = time.time()
    print(t2-t1)


    t3 = time.time()
    print(A.merge2(nums1, m, nums2, n))
    t4 = time.time()
    print(t4-t3)



