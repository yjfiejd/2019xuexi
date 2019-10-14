# @TIME : 2019/9/9 下午10:29
# @File : 合并区间_排序_56.py
"""
给出一个区间的集合，请合并所有重叠的区间。
示例 1:
输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
示例 2:

输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
"""

class Solution:
    """想法很简单，先排序，然后对比第一个元素的[1] 与 第二个元素的[0] 进行合并"""
    def merge(self, intervals):
        if intervals == []:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        rel = [intervals[0]]
        for i in range(1, len(intervals)):
            s1 = rel[-1][1]
            s2 = intervals[i][0]
            s3 = intervals[i][1]
            big = max(s1, s3)
            if s1 >= s2:
                rel[-1][1] = big
            else:
                rel.append(intervals[i])
        return rel



if __name__ == "__main__":
    A = Solution()
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    answ = A.merge(intervals)
    print(answ)


















