# @TIME : 2019/3/30 下午2:14
# @File : 浅拷贝-深拷贝.py



# 对象赋值操作！
print('--对象赋值操作--\n')
will = ["will", 28, ["python", "c#", "JavaScript"]]

wilber = will


print("will的id是",id(will))
print("wilber的id是",id(wilber))
print("will 中元素 id: ", [id(i) for i in will])
print("wilber 中元素 id: ", [id(i) for i in wilber])
print("will  = ", will)
print("wilber = ", wilber)


# 修改元素
will[0] = "HAHA"     #这里需要注意的一点是，str是不可变类型，所以当修改的时候会替换旧的对象
will[2].append("C++")
print('-----------------------------------------------------\n')

print("will的id是",id(will))
print("wilber的id是",id(wilber))
print("will 中元素 id: ", [id(i) for i in will])
print("wilber 中元素 id: ", [id(i) for i in wilber])
print("will  = ", will)
print("wilber = ", wilber)






# 浅拷贝
print("\n--浅拷贝--\n")

import copy

a = ["will", 28, ['python', 'c#', 'JavaScript']]
b = copy.copy(a)
print("a的id是",id(a))
print("b的id是",id(b))
print("a 中元素 id: ", [id(i) for i in a])
print("b 中元素 id: ", [id(i) for i in b])
print("a  = ", a)
print("b = ", b)

a[0] = 'haha'
a[2].append('c++')
print('-----------------------------------------------------\n')
print("a的id是",id(a))
print("b的id是",id(b))
print("a 中元素 id: ", [id(i) for i in a])
print("b 中元素 id: ", [id(i) for i in b])
print("a  = ", a)
print("b = ", b)


# 总结一下，当我们使用下面的操作的时候，会产生浅拷贝的效果：
#
# 使用切片[:]操作
# 使用工厂函数（如list/dir/set）
# 使用copy模块中的copy()函数




# 深拷贝
print("\n--深拷贝--\n")
a = ["will", 28, ['python', 'c#', 'JavaScript']]
b = copy.deepcopy(a)
print("a的id是",id(a))
print("b的id是",id(b))
print("a 中元素 id: ", [id(i) for i in a])
print("b 中元素 id: ", [id(i) for i in b])
print("a  = ", a)
print("b = ", b)

a[0] = 'haha'
a[2].append('c++')
print('-----------------------------------------------------\n')
print("a的id是",id(a))
print("b的id是",id(b))
print("a 中元素 id: ", [id(i) for i in a])
print("b 中元素 id: ", [id(i) for i in b])
print("a  = ", a)
print("b = ", b)


# 其实，对于拷贝有一些特殊情况：
#
# 对于非容器类型（如数字、字符串、和其他'原子'类型的对象）没有拷贝这一说
# 也就是说，对于这些类型，"obj is copy.copy(obj)" 、"obj is copy.deepcopy(obj)"
# 如果元祖变量只包含原子类型对象，则不能深拷贝，看下面的例子