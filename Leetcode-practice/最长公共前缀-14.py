# @TIME : 2019/3/30 下午10:10
# @File : 最长公共前缀-14.py


# class Solution:
#
#     def longestCommonPrefix(self, strs):
#
#         result = ""
#         i = 0
#
#         while True:
#             try:
#                 sets = set(string[i] for string in strs)
#                 if len(sets) == 1:
#                     result += sets.pop()
#                     i += 1
#                 else: break
#             except Exception as e:
#                 break
#         return result

class Solution:

    def longestCommonPrefix(self, strs):

        if len(strs) < 1 or not strs:
            return ""
        if len(strs) == 1:
            return strs[0]

         # 元素全部相同
        if len(set(strs)) == 1:
            return strs[0]

        # 找最短str
        short_i = 0
        temp_len = len(strs[0])
        for i in range(1, len(strs)):
            if len(strs[i]) < temp_len:
                short_i = i

        # 找最短前缀
        short_str = strs[short_i]
        for i in range(len(short_str)):
            for string in strs:
                # print(string[i])
                # print(short_str[i])
                if string[i] != short_str[i]:
                    return short_str[:i]

        return short_str


        # for j in range(1, len(strs)):
        #     while (strs[j].index(short_str) != 0):
        #         short_str = short_str[:len(short_str) - 1]
        #         print(short_str)
        # return short_str


a = Solution()
# rel = a.longestCommonPrefix(["abab","aba","abc"])
# rel = a.longestCommonPrefix(["ca","a"])
# rel = a.longestCommonPrefix(["c","c"])
rel = a.longestCommonPrefix(["aa","a"])
print(rel)
