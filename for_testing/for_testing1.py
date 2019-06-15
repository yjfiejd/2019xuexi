# @TIME : 2019/6/15 下午2:06
# @File : for_testing1.py


# spam = {'name':1, 'age':2}
#
# spam.setdefault('color', 9)
# c = spam.setdefault('name', 10)
#
# print(c)
#
# print(spam)


# spam = {"name":[1], "age":[2]}
# c = spam.setdefault("name", [])
#
# print('c = ', c)
# print('spam=', spam)


b = [1, 2, 3, 2]
new = {}
# for i, elt in enumerate(b):
#     cc = new.setdefault(elt, [])
#     print('\n')
#     print('cc = ',cc)
#     print('new = ', new)
#
#     cc.append(i)
#     print('cc =', cc)
#     print('new = ', new)


new = {'a':1, "b":2, "c":3}

c = new.get('d', [])
print('c = ', c)


