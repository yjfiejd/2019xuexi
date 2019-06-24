# @TIME : 2019/6/18 上午12:28
# @File : for_testing.py

import requests
import urllib.request
import ssl

from bs4 import BeautifulSoup

ssl._create_default_https_context = ssl._create_unverified_context

# url = "http://www.csres.com/notice/50655.html"
url = "https://www.wordbee-translator.com/a/andovar/Jobs/JobDetailsView.aspx?x=JOqem1TmNel2rk3zMUEkdco9rUhSPnhkE4QIyq6lSDReDHCndQmft21kH7zIVlPqvzR6JFS2KT4%3d"
rel = requests.get(url)

# soup = BeautifulSoup(rel, 'html.parser')
a = 1