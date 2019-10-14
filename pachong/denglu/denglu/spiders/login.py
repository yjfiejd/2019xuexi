# -*- coding: utf-8 -*-
import scrapy
import logging
from scrapy.utils.response import open_in_browser


from scrapy import Request, FormRequest


class LoginSpider(scrapy.Spider):
    name = 'login'
    allowed_domains = ['auth.dxy.cn']
    # start_urls = ['https://auth.dxy.cn/accounts/login']
    start_urls = ['https://auth.dxy.cn/accounts/login?qr=false&method=1']

    def parse(self, response):
        token = response.css('div input::attr(value)').extract()[-2]
        print('token = ', token)

        return FormRequest.from_response(response, formdata={
            'loginType': '1',
            'username': 'xxxxx',
            'password': 'xxxxx',
            'keepOnlineType': '2',
            'trys': '0',
            'nlt': token,
            '_eventId': 'submit',
        }, callback = self.parsing_2)

    def parsing_2(self, response):
        if response.status == 200:
            print('status = ', response.status)
            # print('html = ', response.text)
            next_url = response.css('div a::attr(href)').extract_first()
            print("next_url = ", next_url)
            yield scrapy.Request(url=next_url, callback = self.parsing_drug, dont_filter = True)

    def parsing_drug(self, response):
        # open_in_browser(response)
        print('drug response = ', response.status)
        if response.status == 200:
            drug_url = response.css('div ul li a[title="用药助手"]::attr(href)').extract_first()
            print("drug_url = ", drug_url)
            yield scrapy.Request(url=drug_url, callback= self.drugs, dont_filter=True)

    def drugs(self, response):
        print('drugs response = ', response.status)
        if response.status == 200:
            single = "http://drugs.dxy.cn/drug/52424.htm"
            yield scrapy.Request(url=single, callback= self.single, dont_filter=True)

    def single(self, response):
        print('single response = ', response.status)
        # print(response.text)
        dd = response.css('dd::text').extract()
        print(dd)
        # print(response.text)
