# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import FormRequest
from scrapy.utils.response import open_in_browser

# from pachong.denglu.denglu.items import DengluItem
from ..items import DengluItem


class QuoteSpider(scrapy.Spider):
    name = 'quote'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com/login']

    def parse(self, response):
        # open_in_browser(response)
        token = response.css('form input::attr(value)').extract_first()
        print(token)

        return FormRequest.from_response(response, formdata={
            "csrf_token":token,
            "username":'xxxx',
            "password":'xxxx'
        }, callback = self.parsing_2)

    def parsing_2(self, response):
        item = DengluItem()

        total = response.css('div.quote')
        # print('total = ', total)

        for quote in total:
            title = quote.css('span.text::text').extract()

            item['text'] = title
            yield  item