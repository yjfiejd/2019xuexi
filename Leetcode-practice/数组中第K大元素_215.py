# @TIME : 2019/10/9 下午01:04
# @File : 数组中第K大元素_215.py


"""
在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
说明:

你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。

"""
import heapq
import time

b = [3,2,1,5,6,4]


class Solution:

    def findKthLargest(self, nums, k):
        heap = nums
        # print(heap)
        self.heapsort(heap)
        # print(heap)

        # new = []
        # for i in heap:
        #     if i not in new:
        #         new.append(i)
        new = heap[::-1]
        print(new)
        find = new[k-1]
        return find

    def max_heapify(self, heap, heapsize, root):
        """堆调整"""
        left = 2*root + 1
        right = left + 1
        larger = root

        if left < heapsize and heap[larger] < heap[left]:
            larger = left

        if right < heapsize and heap[larger] < heap[right]:
            larger = right

        if larger != root:
            heap[larger], heap[root] = heap[root], heap[larger]
            self.max_heapify(heap, heapsize, larger)

    def build_max_heap(self, heap):
        """构造一个堆, 未进行排序"""
        heapsize = len(heap)
        for i in range((heapsize-2)//2, -1, -1):  # 从后往前调整
            self.max_heapify(heap, heapsize, i)

    def heapsort(self, heap):
        """将根节点取出与最后一位做对调，对前面len-1个节点继续进行对调整过程"""
        self.build_max_heap(heap)
        for i in range(len(heap) - 1, -1, -1):
            heap[0], heap[i] = heap[i], heap[0]
            self.max_heapify(heap, i, 0)  # size 不断减小，这里是i



class Solution2:
    def findKthLargest(self, nums, k):
        return heapq.nlargest(k, nums)[-1]







if __name__ == "__main__":


    # nums = [3,2,1,5,6,4]
    # k = 2
    nums = [3,2,3,1,2,4,5,5,6]
    k = 4

    t1 = time.time()
    S = Solution()
    find = S.findKthLargest(nums, k)
    t2 = time.time()

    print(find)
    print("time1:",t2-t1)

    t3 = time.time()
    S2 = Solution2()
    find2 = S2.findKthLargest(nums, k)
    t4 = time.time()

    print(find2)
    print("time2:",t4-t3)
