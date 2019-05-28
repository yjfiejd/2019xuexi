# @TIME : 2019/4/26 下午11:45
# @File : 平方值多少种取值.py


# 给你一个有序整数数组，数组中的数可以是正数、负数、零，请实现一个函数，这个函数返回一个整数：返回这个数组所有数的平方值中有多少种不同的取值。
# 举例：
# nums = {-1,1,1,1},那么你应该返回的是：1。因为这个数组所有数的平方取值都是1，只有一种取值
# nums = {-1,0,1,2,3}你应该返回4，因为nums数组所有元素的平方值一共4种取值：1,0,4,9





def get_sqrt_cate(nums):

    nums = [abs(i)*abs(i) for i in nums]
    nums2 = list(set(nums))
    return len(nums2)


nums = {-1,0,1,2,3}
nums = {-1,1,1,1}

len1 = get_sqrt_cate(nums)
a = 1































