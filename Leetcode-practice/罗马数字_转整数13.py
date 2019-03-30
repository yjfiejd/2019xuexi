# @TIME : 2019/3/30 下午4:01
# @File : 罗马数字_转整数13.py

import time

def romanToInt(s):
    roman_dict = {"I": 1,
                  "V": 5,
                  "X": 10,
                  "L": 50,
                  "C": 100,
                  "D": 500,
                  "M": 1000,
                  }
    two_roman_dict = {
                  "IV": 4,
                  "IX": 10,
                  "XL": 40,
                  "XC": 90,
                  "CD": 400,
                  "CM": 900
                  }

    add_item = []

    def get_sum(k, v, s):
        if k in s:
            how_many = s.count(k)
            s = s.replace(k, "")
            if how_many > 1:
                add_item.append(v*how_many)
            else:
                add_item.append(v)
        return add_item, s

    # 先替换长的子串，然后加入list, 统计一下出现的次数
    for k, v in two_roman_dict.items():
        add_item, s = get_sum(k, v, s)

    # 短子串
    for m, n in roman_dict.items():
        add_item, s = get_sum(m, n, s)

    return sum(add_item)


start = time.time()

a = romanToInt("MCMXCIV")
end = time.time()
print('第一个程序运行时间：', str(end - start))

print("结果：", a)



# 别人的解法二：
# 思路：因为都是大的数字在前面，所以利用，如果出现前面数字大于后面的，则它是组合，
# 注意，需要减去中间的那个重复的项

def romantoint(s):

    roman_dict = {"I": 1,
                  "V": 5,
                  "X": 10,
                  "L": 50,
                  "C": 100,
                  "D": 500,
                  "M": 1000,
                  }
    result = 0
    for i in range(len(s)):

        if i > 0 and roman_dict[s[i]] > roman_dict[s[i-1]]:
            result += roman_dict[s[i]] - 2*roman_dict[s[i-1]]
        else:
            result += roman_dict[s[i]]

    return result



start2 = time.time()

b = romantoint("MCMXCIV")
end2 = time.time()
print('第二个程序运行时间：', str(end2 - start2))

print("结果：", b)

0 + M + C + (M-C) + X

0 + M + C + M