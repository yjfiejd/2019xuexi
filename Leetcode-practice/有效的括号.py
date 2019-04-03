# @TIME : 2019/4/2 上午8:36
# @File : 有效的括号.py


class Solution:
    def isValid(self, s: str) -> bool:
        ok_dict = {
            "1": ["(", ")"],
            "2": ["{", "}"],
            "3": ["[", "]"]
        }

        # 从左到有，搜索，第一个属于哪个集合


a = Solution()
a.isValid("(){}")




lxc2530366
