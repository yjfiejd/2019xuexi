# @TIME : 2019/7/29 下午10:42
# @File : tire_tree.py

class TrieNode:

    def __init__(self):
        self.children = [None] * 26
        self.isEndOfWord = False

class Tire:

    def __init__(self):
        self.root = self.getNode()

    def getNode(self):
        return TrieNode()

    def _charToIndex(self, ch):
        """Converts key current character into index
        # use only 'a' through 'z' and lower case """
        rel = ord(ch) - ord('a')
        return rel

    def insert(self, key):
        pCrawl = self.root  # 头指针还保留着, 这个设计不错
        length = len(key)
        for level in range(length):
            a = key[level]
            index = self._charToIndex(key[level])
            aa = pCrawl.children[index]
            if not pCrawl.children[index]:
                pCrawl.children[index] = self.getNode()

            pCrawl = pCrawl.children[index]  # 像LinkList的节点移动

        pCrawl.isEndOfWord = True

    def search(self, key):
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            a = key[level]
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                return False
            pCrawl = pCrawl.children[index]

        return pCrawl != None and pCrawl.isEndOfWord


if __name__ == "__main__":

    keys = ['the', 'a', 'there', 'anaswe', 'any', 'by', 'their']

    t = Tire()

    for key in keys:
        t.insert(key)

    a = 1
    print(t.search('there'))


