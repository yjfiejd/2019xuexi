# @TIME : 2019/3/17 下午10:52
# @File : chainer_implementation.py


# 总体结构
class My_CRF():

    def __init__(self):
        """
        randomly initialize transition score
        """

    def __call__(training_data_set):
        # [ Loss Function]
        Total_Cost = 0.0

        # Compute CRF Loss
        """
        for sentence in training_data_set:
            1) The real path score of current sentence according the true labels
            2) The log total score of all the possible paths of current sentence
            3) Compute the cos on this sentence using result 1) and 2)
            4) Total_Cost += Cost of this sentence
        """

        return Total_Cost

    def argmax(new_sentences):
        # [Prediction]
        """
        Predict labels for new sentences
        """