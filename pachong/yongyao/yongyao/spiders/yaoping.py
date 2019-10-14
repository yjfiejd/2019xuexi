# -*- coding: utf-8 -*-
import scrapy


class YaopingSpider(scrapy.Spider):
    name = 'yaoping'
    allowed_domains = ['drugs.dxy.cn']
    start_urls = ['https://drugs.dxy.cn/drug/89095.htm/']

    def parse(self, response):

        pass
