# @TIME : 2019/8/4 下午12:19
# @File : 两数之和.py
import time

"""
nums = [2, 7, 11, 15]
target = 9

return [0, 1] index
"""


nums = [2, 7, 11, 15]
target = 13

# 方法一：
def twosum1(nums, target):
    """暴力方法: O(n^2)"""
    for i in range(len(nums)-1):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

s1 = time.time()
answ = twosum1(nums, target)
e1 = time.time()
print(answ)
print("time:",e1 - s1)


# 方法二：
def twosum2(nums, target):
    """用一次循环 + for的同时在list后部查找, 有可能最差情况sec_num的在最后那复杂度是O(n)"""
    for i in range(len(nums) - 1):
        sec_num = target - nums[i]
        if sec_num in nums[i:]:
            sec_indx = nums[i:].index(sec_num)
            a = 1
            return [i, sec_indx + i]

print('*'*20)
s1 = time.time()
answ = twosum2(nums, target)
e1 = time.time()
print(answ)
print("time:",e1 - s1)


# 方法三:
def twosum3(nums, target):
    """使用哈希表存储 -> map -> dict"""
    mapping = dict()
    for i in range(len(nums)):
        sec_num = target - nums[i]
        if sec_num in mapping:
            return [mapping[sec_num], i]
        else:
            mapping[nums[i]] = i


print('*'*20)
s1 = time.time()
answ = twosum3(nums, target)
e1 = time.time()
print(answ)
print("time:",e1 - s1)


# 方法四：
def binary_search(nums, target):
    l = 0
    r = len(nums)-1
    while l <= r:
        mid = int((l + r) / 2)
        if target == nums[mid]:
            return mid
        elif target < nums[mid]:
            r = mid -1
        elif target > nums[mid]:
            l = mid + 1
    return -1

def twosum4(nums, target):
    """使用 切片后的二分查找 代替哈希，或者是list后半部分切片查找"""
    for i in range(len(nums)):
        sec_num = target - nums[i]
        binary_idx = binary_search(nums[i:], sec_num)
        if binary_idx != -1:
            return [i, binary_idx+i]


print('*'*20)
s1 = time.time()
answ = twosum4(nums, target)
e1 = time.time()
print(answ)
print("time:",e1 - s1)
