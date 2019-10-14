# @TIME : 2019/10/10 上午7:25
# @File : 堆排序.py


"""
堆排序，顾名思义，就是基于堆。因此先来介绍一下堆的概念。 
堆分为最大堆和最小堆，其实就是完全二叉树。最大堆要求节点的元素都要大于其孩子，最小堆要求节点元素都小于其左右孩子，两者对左右孩子的大小关系不做任何要求，其实很好理解。有了上面的定义，我们可以得知，处于最大堆的根节点的元素一定是这个堆中的最大值。其实我们的堆排序算法就是抓住了堆的这一特点，每次都取堆顶的元素，将其放在序列最后面，然后将剩余的元素重新调整为最大堆，依次类推，最终得到排序的序列。

通常堆是通过一维数组来实现的。在阵列起始位置为0的情况中
(1)父节点i的左子节点在位置(2*i+1);
(2)父节点i的右子节点在位置(2*i+2);
(3)子节点i的父节点在位置floor((i-1)/2);
"""

"""
MAX_Heapify: 最大堆调整，将堆的末端子节点作调整，使得子节点永远小于父节点。这是核心步骤，在建堆和堆排序都会用到。比较i的根节点和与其所对应i的孩子节点的值。当i根节点的值比左孩子节点的值要小的时候，就把i根节点和左孩子节点所对应的值交换，当i根节点的值比右孩子的节点所对应的值要小的时候，就把i根节点和右孩子节点所对应的值交换。然后再调用堆调整这个过程，这是一个递归的过程。

Build_Max_Heap: 将堆所有数据重新排序， 建立堆的过程就是不断做最大堆调整的过程，从

HeapSort: 堆排序，移除位在第一个数据的根节点，并做最大堆调整的递归运算。堆排序是利用建堆和堆调整来进行的。首先先建堆，然后将堆的根节点选出与最后一个节点进行交换，然后将前面len-1个节点继续做堆调整的过程。直到将所有的节点取出，对于n个数我们只需要做n-1次操作。
"""

import random

def max_heapify(heap, heapsize, root):
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
        max_heapify(heap, heapsize, larger)

def build_max_heap(heap):
    """构造一个堆"""
    heapsize = len(heap)
    for i in range((heapsize-2)//2, -1, -1):  # 从后往前调整
        max_heapify(heap, heapsize, i)

def heapsort(heap):
    """将根节点取出与最后一位做对调，对前面len-1个节点继续进行对调整过程"""
    build_max_heap(heap)
    for i in range(len(heap)-1, -1, -1):
        heap[0], heap[i] = heap[i], heap[0]
        max_heapify(heap, i, 0)  # size 不断减小，这里是i


if __name__ == "__main__":
    a = [30, 50, 57, 77, 62, 78, 94, 80, 84]
    print(a)

    heapsort(a)
    print(a)
