# @TIME : 2019/6/24 下午9:51
# @File : plus_one_66.py


class Solution:
    # def plusOne(self, digits):
    #
    #     old = int("".join([str(i) for i in digits]))
    #     last_num = digits.pop()
    #     last_num = str(last_num)
    #
    #     if "9" not in last_num:
    #         digits.append(int(last_num)+1)
    #         return digits
    #     else:
    #         old = old+1
    #         new_nums = [int(i) for i in  list(str(old))]
    #         return new_nums

    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        if len(digits) == 0:
            digits = [1]
        elif digits[-1] == 9:
            digits = self.plusOne(digits[:-1])
            digits.extend([0])
        else:
            digits[-1] += 1
        return digits

a = Solution()
bb = a.plusOne([1, 2, 3, 9])
# bb = a.plusOne([199])
print(bb)