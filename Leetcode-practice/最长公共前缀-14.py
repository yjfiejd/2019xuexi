# @TIME : 2019/3/30 下午10:10
# @File : 最长公共前缀-14.py


a = ["flower", "flow", "flight"]

def longestCommonPrefix(strs):
    """
    :param strs:
    :return: 最长公共前缀
    """

    if not strs:
        return ""
    s1 = min(strs)
    s2 = max(strs)

    for i, x in enumerate(s1):
        if x != s2[i]:
            return s2[:i]
    return s1



b = longestCommonPrefix(a)
print(b)
