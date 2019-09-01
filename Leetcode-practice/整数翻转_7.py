# @TIME : 2019/8/19 下午9:46
# @File : 整数翻转_7.py

import time

"""
给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

示例 1:

输入: 123
输出: 321
 示例 2:

输入: -123
输出: -321
示例 3:

输入: 120
输出: 21

注意:

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。

"""

import math
class Solution:
    def reverse(self, x: int) -> int:
        x = str(x)
        ss = math.pow(-2, 31)
        bb = math.pow(2, 31) -1

        if x == "0":
            return 0

        if "0" not in x and "-" not in x:
            new = int(x[::-1])
            if new >= ss and new <= bb:
                return int(new)
            else: return 0

        elif "0" in x and "-" not in x:
            if x[-1] != "0":
                new = int(x[::-1])
                if new >= ss and new <= bb:
                    return int(new)
                else:
                    return 0
            else:
                x = x[:-1]
                new = int(x[::-1])
                if new >= ss and new <= bb:
                    return int(new)
                else:
                    return 0

        elif "-" in x and "0" not in x:
            x = x[1:]
            x = "-" + x[::-1]
            new = int(x)
            if new >= ss and new <= bb:
                return int(new)
            else: return 0

        elif "0" in x and "-" in x:
            if x[-1] != "0":
                x = x[1:]
                x = "-" + x[::-1]
                new = int(x)
                if new >= ss and new <= bb:
                    return int(new)
                else:
                    return 0
            else:
                x = x[:-1]
                x = x[1:]
                x = "-" + x[::-1]
                new = int(x)
                if new >= ss and new <= bb:
                    return int(new)
                else:
                    return 0


class Solution2:
    """Solution 写的再精简一点"""
    def reverse(self, x: int) -> int:
        num = abs(x)
        num = str(num)[::-1]
        if x > 0 and int(num) <= 2 ** 31 - 1:
            return int(num)
        elif x < 0 and int(num) < 2** 31:
            return -int(num)
        else:
            return 0


class Solution3:
    """利用 %10 取最后一位，利用 /10 向左移动"""
    def reverse(self, x: int) -> int:
        num = 0
        abs_a = abs(x)
        while (abs_a != 0):
            temp = abs_a % 10
            num = temp + num * 10
            abs_a = int(abs_a / 10)
        if x > 0 and num < 2 ** 31 - 1:
            return num
        elif x < 0 and num < 2** 31:
            return -num
        else:
            return 0



if __name__ == "__main__":

    start = time.time()
    A = Solution()
    rel = A.reverse(0)
    print(rel)
    rel = A.reverse(123)
    print(rel)
    rel = A.reverse(-123)
    print(rel)
    rel = A.reverse(120)
    print(rel)
    rel = A.reverse(-12300)
    print(rel)
    rel = A.reverse(123829734932562881624712836)
    print(rel)
    end = time.time()
    print("time:", end-start)


    start = time.time()
    A = Solution2()
    rel = A.reverse(0)
    print(rel)
    rel = A.reverse(123)
    print(rel)
    rel = A.reverse(-123)
    print(rel)
    rel = A.reverse(120)
    print(rel)
    rel = A.reverse(-12300)
    print(rel)
    rel = A.reverse(123829734932562881624712836)
    print(rel)
    end = time.time()
    print("time:", end-start)

