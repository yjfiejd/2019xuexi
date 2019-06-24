# @TIME : 2019/6/15 上午9:21
# @File : my_SequenceMatcher.py
# ----------------------------------------------------------------

# 学习来自python库, diff
# -> SequenceMatch: (未使用二维数组,使用一维数组,字典，双指针over)
# -> 还可以指定 isjunk字符，设置是否跳过

from collections import namedtuple as _namedtuple
import time

Match = _namedtuple('Match', 'a b size')

class SequenceMatcher:

    def __init__(self, isjunk=None, a='', b='', autojunk=True):
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.set_seqs(a, b)
        self.Match = _namedtuple('Match', 'a b size')

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        if a is self.a:
            return
        self.a = a

    def set_seq2(self, b):
        """新建dict ->  存储b序列中每个元素对应的所有位置信息{value:[value_idx1.. value_idx2..]}"""
        if b is self.b:
            return
        self.b = b
        self.__chain_b()

    def __chain_b(self):
        """双下划线内部函数 -> 构建一会用的字典"""
        b = self.b
        self.b2j = b2j = {}

        for idx, v in enumerate(b):
            # 这个indices 是对应的 v 取出来的值，没有的话默认 []
            indices = b2j.setdefault(v, [])  #如果b2j字典中没有v这个key，那就设置一波默认值[], 如果有取出它的value，返回给indices，用来添加idx
            indices.append(idx)

        self.bjunk = junk = set()
        isjunk = self.isjunk
        # 如果存在有junk字典，需要先把他们从b2j中删除
        if isjunk:
            for elt in b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk:
                del b2j[elt]

        # 清除popular元素
        self.bpopular = popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt, idxs in b2j.items():
                if len(idxs) > ntest:
                    popular.add(elt)

            for elt in popular:
                del b2j[elt]

    def find_longest_match(self, a, b):
        alo = 0
        ahi = len(a)
        blo = 0
        bhi = len(b)

        a, b, b2j, isbjunk = self.a, self.b, self.b2j, self.bjunk.__contains__
        besti, bestj,bestsize = alo, blo, 0
        j2len = {}  # 有点像用二维数组动态规划解法中的k，记录第几次连续成功匹配的值
        nothing = []  # 当b2j中，也就是b中没有a中存在的元素时候，返回空

        for i in range(alo, ahi):
            j2lenget = j2len.get   #获得j2len字典对象中的方法
            newj2len = {}
            for j in b2j.get(a[i], nothing):
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = newj2len[j] = j2lenget(j-1, 0) + 1  # 如果b中某个字符(idx=j)匹配成功，看下上一个j中的len
                if k > bestsize:
                    besti, bestj, bestsize = i-k+1, j-k+1, k  # 更新指针位置，i, j 需要先减去之前的k + 1 (配上了一个)

            j2len = newj2len

        #  后面可以先省略，这里没有用到junk
        while besti > alo and bestj > blo and \
              not isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              not isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1

        while besti > alo and bestj > blo and \
              isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize = bestsize + 1

        # 可以自己控制输入
        a = 1
        return Match(besti, bestj, bestsize)



if __name__ == "__main__":

    A = [1,2,3,2,1]
    B = [3,2,1,4,7]

    rels = []
    for i in range(100):
        start = time.time()
        # for i in range(100):
        m = SequenceMatcher(None, A, B)
        end = time.time()
        temp = end-start
        rels.append(temp)

    print("average_time:",sum(rels)/100)
    # print(m.find_longest_match(A, B))



