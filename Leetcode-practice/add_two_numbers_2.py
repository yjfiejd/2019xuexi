# @TIME : 2019/7/28 下午7:40
# @File : add_two_numbers_2.py

"""
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.

"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def traverse(self):
        node = self
        while node != None:
            print(node.val)
            node = node.next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:

        dummy = ListNode(0)
        pointer = dummy
        carry = 0

        while l1 and l2:

            pointer.next = ListNode((l1.val + l2.val + carry) % 10)  # 取个位数
            carry = (l1.val + l2.val + carry) // 10  # 取整

            # 指针移动
            l1 = l1.next
            l2 = l2.next
            pointer = pointer.next

        # 留下更长的l
        l = l1 if l1 else l2

        while l:
            pointer.next = ListNode((l.val + carry) % 10)
            carry = (l.val + carry) // 10

            # 指针移动
            l = l.next
            pointer = pointer.next

        if carry == 1:  # 最高位溢出注意
            pointer.next = ListNode(1)

        return dummy.next


if __name__ == "__main__":
    # Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    # Output: 7 -> 0 -> 8
    # Explanation: 342 + 465 = 807.

    l1 = ListNode(2)
    b = ListNode(4)
    c = ListNode(3)
    l1.next = b
    b.next = c

    l2 = ListNode(5)
    e = ListNode(6)
    f = ListNode(4)
    l2.next = e
    e.next = f

    A = Solution()
    c = A.addTwoNumbers(l1, l2)
    print('c = ', c)
    c.traverse()




