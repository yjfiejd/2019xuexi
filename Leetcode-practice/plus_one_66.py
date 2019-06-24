# @TIME : 2019/6/24 下午9:51
# @File : plus_one_66.py

from copy import deepcopy

class Solution:
    def plusOne(self, digits):

        old = int("".join([str(i) for i in digits]))
        last_num = digits.pop()
        last_num = str(last_num)

        if "9" not in last_num:
            digits.append(int(last_num)+1)
            return digits
        else:
            old = old+1
            new_nums = [int(i) for i in  list(str(old))]
            return new_nums


a = Solution()
# bb = a.plusOne([1, 2, 3])
bb = a.plusOne([199])
print(bb)