# @TIME : 2019/6/23 下午1:31
# @File : for_6-23.py
import re


path = "/Users/a1/Downloads/pycharm project_new/2019xuexi/for_testing/txt_test"


final = []

with open(path, 'r', encoding='utf-8') as f:
    for line in f.readlines():

        rel = re.findall("\s[\u4e00-\u9fa5].*", line)
        if rel:
            rel = rel[0]
            save = line.replace(rel, "")
            final.append(save)
        else:
            final.append(line)



all_str = "".join(final)
print('all_str = ', all_str)