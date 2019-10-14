# @TIME : 2019/9/14 下午7:44
# @File : githublog.py


import requests
from lxml import etree

data = {
'loginType':'1',
'username':'18321736625',
'password':'lxc2530366',
'keepOnlineType':'2',
'trys':'0',
'nlt':'_c5E9C2E84-8E3D-B255-5E33-4428C2831554_kC5A72A4C-D032-AEC3-974C-2B2E334D8906',
'_eventId':'submit',
'geetest_challenge':'d6ef04271cb888ae0a13233627016608',
'geetest_validate':'67c2bc2f7cd920b2675821d2a8a153ee',
'geetest_seccode':'67c2bc2f7cd920b2675821d2a8a153ee|jordan',
}

post_url = "https://auth.dxy.cn/accounts/login?service=http%3A%2F%2Fdrugs.dxy.cn%2Flogin.do%3Fdone%3Dhttp%253A%252F%252Fdrugs.dxy.cn%252Fdrug%252F121029%252Fdetail.htm&qr=false&method=1"
r = requests.post(url=post_url, data=data)
print(r.text)
a = 1






# class Login(object):
#     def __init__(self):
#         self.headers = {
#             'Host': 'github.com',
#             'Referer': 'https://github.com/login',
#             'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'
#         }
#         self.login_url = 'https://github.com/login'
#         self.post_url = 'https://github.com/session'
#         self.logined_url = 'https://github.com/settings/profile'
#         self.session = requests.Session()
#
#     def token(self):
#         response = self.session.get(self.login_url, headers = self.headers)
#         selector = etree.HTML(response.text)
#         token = selector.xpath('//div//input[2]/@value')
#         return token
#
#     def login(self, email, password):
#         post_data = {
#             'commit':'Sign in',
#             'utf8':'✓',
#             'authenticity_token': self.token()[0],
#             'login': email,
#             'password': password
#         }
#         response = self.session.post(self.post_url, data=post_data, headers = self.headers)
#         print(response)
#         a = 1
#         if response.status_code == 200:
#             self.dynamics(response.text)
#
#     def dynamics(self, html):
#         selector = etree.HTML(html)
#         dynamics= selector.xpath('//div//div//div//p')
#     #     dynamics = selector.xpath('//div[contains(@class, "news")]//div[contains(@class, "alert")]')
#     #     for item in dynamics:
#     #         dynamic = ' '.join(item.xpath('.//div[@class="title"]//text()')).strip()
#     #         print("dynamic = ", dynamic)
#     #
#     # def profile(self, html):
#     #     selector = etree.HTML(html)
#     #     name = selector.xpath('//input[@id="user_profile_name"]/@value')[0]
#     #     email = selector.xpath('//select[@id="user_profile_email"]/option[@value!=""]/text()')
#     #     print("name = ", name, "email = ", email)
#
#
#
# if __name__ == "__main__":
#     login = Login()
#     email = 'yjfiejd@163.com'
#     password = '@lxc2530366'
#     login.login(email=email, password=password)



