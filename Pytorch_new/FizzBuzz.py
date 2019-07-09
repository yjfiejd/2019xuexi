# @TIME : 2019/7/2 上午9:00
# @File : FizzBuzz.py

# 写神经网络 玩这个游戏。

def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i% 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def helper(i):
    return fizz_buzz_decode(i, fizz_buzz_encode(i))



for i in range(1, 16):
    print(helper(i))

