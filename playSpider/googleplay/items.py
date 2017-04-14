# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class GoogleplayItem(scrapy.Item):

    app_title = scrapy.Field()
    description = scrapy.Field()
    similar_apps = scrapy.Field()
    category = scrapy.Field()
