# @TIME : 2019/3/30 下午1:28
# @File : palindrome_num_09.py
import copy


import time

def palindrome_num(num):

    # 正数
    # if num >0:
    src = str(num)
    src_list = [i for i in src]
    temp = copy.deepcopy(src_list)
    temp.reverse()
    a = 1
    for i in range(len(src_list)):
        if src_list[i] == temp[i]:
            continue
        else:
            return False

    return True



def isPalindrome(x) -> bool:

    num = 0
    a = abs(x)

    while (a != 0):
        temp = a % 10
        num = num * 10 + temp
        a = a // 10

    if x >= 0 and x == num:
        return True
    else:
        return False

start = time.time()
aa = palindrome_num(1222)
end = time.time()
print('第一个程序运行时间：', str(end - start))
print('结果：', aa)




start2 = time.time()
bb = isPalindrome(1222)
end2 = time.time()
print('第二个程序运行时间：', str(end2 - start2))
print('结果：', bb)


