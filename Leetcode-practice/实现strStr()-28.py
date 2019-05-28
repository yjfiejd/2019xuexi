# @TIME : 2019/4/12 上午8:05
# @File : 实现strStr()-28.py

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if haystack:
            return haystack.find(needle)
        else:
            return 0












if __name__ == "__main__":

    haystack = "hello"
    needle = "ll"

    haystack = "aaaaa"
    needle = "bba"
    a = Solution()
    b = a.strStr(haystack, needle)

    print('b = ', b)
