# @TIME : 2019/6/15 上午9:21
# @File : my_SequenceMatcher.py

# from difflib import SequenceMatcher
from heapq import nlargest as _nlargest
from collections import namedtuple as _nametuple

Match = _nametuple('Match', 'a b size')

class SequenceMatcher:

    def __init__(self, isjunk=None, a='', b='', autojunk=True):
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.set_seq(a, b)

    def set_seq(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        if a is self.a:
            return
        self.a = a

    def set_seq2(self, b):
        if b is self.b:
            return
        self.b = b
        self.__chain_b()

    def __chain_b(self):
        b = self.b
        self.b2j = b2j = {}

        # build value-index dict
        for i, elt in enumerate(b):
            indices = b2j.setdefault(elt, [])
            indices.append(i)

        # find junk item if has isjunk  -> del junk item
        self.bjunk = junk = set()
        isjunk = self.isjunk
        if isjunk:
            for elt in b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk:
                del b2j[elt]

        # purge popular elements that are not junk
        self.bpopular = popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt, idx in b2j.items():
                if len(idx) > ntest:
                    popular.add(elt)
            for elt in popular:
                del b2j[elt]

    def find_longest_match(self, alo, ahi, blo, bhi):

        a, b, b2j, isbjunk = self.a, self.b, self.b2j, self.bjunk.__contains__
        besti, bestj, bestsize = alo, blo, 0

        j2len = {}   # key是b中与a相匹配的元素，values是代表一共有多少个连续的相同元素了
        nothing = []

        for i in range(alo, ahi):  # 从a中的第 i 个元素遍历
            j2len_i = j2len  # key是 b 中的第j个元素, j记录位置，同时递加前一个值，如果存在的话
            j2lenget = j2len.get
            newj2len = {}
            key_temp = a[i]
            bb = b2j.get(key_temp, nothing)   # 获取 b2j (value:[position]) 中的位置, 存在key返回对应value，否则返回nothing
            blo_i = blo
            bhi_i = bhi
            for j in bb:
                if j < blo:
                    continue
                if j >= bhi:
                    break
                pp = j2lenget(j-1, 0)
                k = newj2len[j] = pp + 1
                if k > bestsize:
                    besti, bestj, bestsize = i-k+1, j-k+1, k
            j2len = newj2len

        junk_judge = isbjunk(b[bestj-1])
        besti_1 = besti
        bestj_1 = bestj
        aaa = a[besti-1]
        bbb = b[bestj-1]
        cc = 1

        while besti > alo and bestj > blo and not isbjunk(b[bestj-1]) and \
            a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1

        while besti+bestsize < ahi and bestj+bestsize < bhi and not isbjunk(b[bestj+bestsize]) and \
            a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1

        while besti > alo and bestj > blo and isbjunk(b[bestj-1]) and a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1


        while besti+bestsize < ahi and bestj+bestsize < bhi and isbjunk(b[bestj+bestsize]) and \
            a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1

        return Match(besti, bestj, bestsize)




# A = [1, 3, 4, 5]
# B = [2, 3, 5, 1, 3]

A = "abcdefghi"
B = "xxxghxxbcdex"





m = SequenceMatcher(None, A, B)
cc = m.find_longest_match(0, len(A), 0, len(B))
print(cc.a)
print(cc.b)
print(cc.size)










































