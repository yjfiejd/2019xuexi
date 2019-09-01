# @TIME : 2019/8/8 上午2:34
# @File : 基本计算器_224.py

import time

"""
实现一个基本的计算器来计算一个简单的字符串表达式的值。

字符串表达式可以包含左括号 ( ，右括号 )，加号 + ，减号 -，非负整数和空格  。

示例 1:

输入: "1 + 1"
输出: 2
示例 2:

输入: " 2-1 + 2 "
输出: 3
示例 3:

输入: "(1+(4+5+2)-3)+(6+8)"
输出: 23
说明：

你可以假设所给定的表达式都是有效的。
请不要使用内置的库函数 eval。

"""


class Solution:
    def calculate(self, s):


        ops = ["+", "-"]
        str1 = s
        str1 = str1.replace(" ", "")

        if self.is_number(str1):
            return int(str1)

        if len(str1) == 1:
            return int(str1)

        if len(str1) == 3 and str1[-1] == ")":
            return int(str1.lstrip("(").rstrip(")"))


        nums = []
        operators = []

        rel = []
        for index in range(len(str1)):
            char = str1[index]
            if char == "(":
                rel.append(char)

            elif char in ops:
                rel.append(char)

            elif char.isnumeric():
                rel.append(char)

            elif char == ")":
                while char != "(":
                    char = rel.pop()
                    if char.replace("-", "").isnumeric():
                        nums.append(char)
                    else:
                        operators.append(char)

                sub_rel = self.cal_func(nums, operators, str1)
                sub_rel = str(sub_rel)
                rel.append(sub_rel)
                a = 1

        if len(rel) > 1:
            while rel:
                char = rel.pop()
                if char.replace("-", "").isnumeric():
                    nums.append(char)
                else:
                    operators.append(char)
            final_rel = self.cal_func(nums, operators, str1)
            return final_rel

        else:
            return int(rel.pop())


    def cal_func(self, nums, operators, str1):
        if operators[-1] == "(":
            operators.pop()
            operators.append("+")

        if "(" not in str1:
            operators.append("+")

        if len(nums) - len(operators) == 1:
            operators.append("+")

        sum = 0
        while operators:
            op = operators.pop()
            if op == "+":
                sum += int(nums.pop())
            elif op == "-":
                sum -= int(nums.pop())
        return sum

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False


s = time.time()

aa = ["1-11" ,"(1+(4+5+2)-3)+(6+8)", "1 + 1", " 2-1 + 2 ", " 2", "(1)", "0  ",  "(4+9)",  "(1-(3-4))", "(1+(4+5+2)-3)+(6+8)", "2147483647"]

A = Solution()
b = {str1: A.calculate(str1) for str1 in aa}
for k, v in b.items():
    print('key:', k, '--->', v)
e = time.time()
print('\ntime:', e - s)





















