# @TIME : 2019/9/8 下午4:48
# @File : for_testing1.py
import json

import requests
import re
from requests.exceptions import RequestException
from requests_toolbelt.threaded.pool import Pool


def get_one_page(url):

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        return None


def parse_one_page(html):
    # pattern = re.compile('<dd>.*?board-index.*?>(\d+)</i>.*?\stitle=(.*?).*', re.S)
    pattern = re.compile(r'title=\"(.*?)\"\s', re.S)
    item = re.findall(pattern, html)
    items = list(set(item))
    items = enumerate(items)
    aa = []

    for idx,  name in items:
        bb = {}
        bb["id"] = idx
        bb["name"] = name
        aa.append(bb)
    return aa

def write(content):
    with open('result.txt', 'a', encoding='utf-8') as f:
        a = json.dumps(content, ensure_ascii=False)
        f.write(a + '\n')



def main(offset):
    url = "https://maoyan.com/board/4?offset={}".format(str(offset))
    html = get_one_page(url)
    # print(html)
    for item in parse_one_page(html):
        print(item)
        write(item)


if __name__ == "__main__":
    # for i in range(10):
    #     main(i*10)
    pass
    # pool = Pool()
    # pool.join_all(main, [i*10 for i in range(10)])



