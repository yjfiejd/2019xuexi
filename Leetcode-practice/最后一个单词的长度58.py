# @TIME : 2019/4/24 上午9:05
# @File : 最后一个单词的长度58.py


class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.rstrip().split(' ')[-1])






class Solution3:
    def lengthOfLastWord(self, s: str) -> int:

        count = 0
        local_count = 0

        for i in range(len(s)):
            if s[i] == ' ':
                local_count = 0
            else:
                local_count += 1
                count = local_count

        return count





a = Solution3()
print(a.lengthOfLastWord(" world "))
