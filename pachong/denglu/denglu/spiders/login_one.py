# -*- coding: utf-8 -*-
import scrapy


class LoginOneSpider(scrapy.Spider):
    name = 'login_one'
    allowed_domains = ['drugs.dxy.cn/drug/133657/detail.htm']
    start_urls = ['http://drugs.dxy.cn/drug/133657/detail.htm/']


    def parse(self, response):
        # dd = response.css('dd::text').extract()

        button = response.css('dt a::attr(href)').extract()
        button_num = [i.lstrip("#") for i in button]

        print(button)
        print(button_num)

        page_num = '133657'







        pass






