# @TIME : 2019/6/11 下午11:54
# @File : sequence_matcher.py


from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple


Match = _namedtuple('Match', 'a b size')


One = Match("hehe", 'sdfdf', 9)

print(One.Match)