# @TIME : 2019/10/11 下午10:11
# @File : 数据流中的第K大元素_703.py


"""
题目：
设计一个找到数据流中第K大元素的类（class）。注意是排序后的第K大元素，不是第K个不同的元素。

你的 KthLargest 类需要一个同时接收整数 k 和整数数组nums 的构造器，它包含数据流中的初始元素。每次调用 KthLargest.add，返回当前数据流中第K大的元素。

示例:

int k = 3;
int[] arr = [4,5,8,2];
KthLargest kthLargest = new KthLargest(3, arr);
kthLargest.add(3);   // returns 4
kthLargest.add(5);   // returns 5
kthLargest.add(10);  // returns 5
kthLargest.add(9);   // returns 8
kthLargest.add(4);   // returns 8
说明:
你可以假设 nums 的长度≥ k-1 且k ≥ 1。
"""


class KthLargest:

    def __init__(self, k, nums):
        pass


    def max_heapify(self, heap, heapsize, root):
        left = 2*root + 1
        right = left + 1
        larger = root

        if left < heapsize and heap[left] > heap[larger]:
            larger = left
        if right < heapsize and heap[right] > heap[larger]:
            larger = right

        if larger != root:
            heap[larger], heap[root] = heap[root], heap[larger]
            self.max_heapify(heap, heapsize, root)

    def build_max_heap(self, heap):
        heapsize = len(heap)
        for i in range((heapsize-2)//2, -1, -1):
            self.max_heapify(heap, heapsize, i)

    def


    def add(self, val):
        pass












