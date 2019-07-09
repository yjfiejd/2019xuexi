# @TIME : 2019/6/25 上午9:02
# @File : remove_duplicates_sorted_list_83.py


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        p = head
        while p != None and p.next != None:
            if p.val == p.next.val:
                p.next = p.next.next
            else:
                p = p.next
        return head

