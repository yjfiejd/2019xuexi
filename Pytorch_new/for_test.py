# @TIME : 2019/7/14 下午5:30
# @File : for_test.py


from abc import ABC
class Promotion(ABC):

    @classmethod
    def discount(self, order):
        """这是返回的折扣金额"""


class FildelityPormo(Promotion):

    def discount(self, order):
        return 2



