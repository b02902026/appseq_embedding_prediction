import scrapy
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from pprint import pprint
from scrapy.utils.response import open_in_browser
from googleplay.items import GoogleplayItem
import logging

logging.log(logging.WARNING, "This is a warning")

class QuotesSpider(scrapy.Spider):
    name = "ggspider"
    start_urls=["https://play.google.com/store/apps?hl=en"]
    test = "test"

    def parse(self,response):

        for url in response.xpath('//a[@class="see-more play-button small id-track-click apps id-responsive-see-more"]'):
            print("len=",len(url.extract()))
            #c_name = url.xpath('a[@class="title-link id-track-click"]/text()')[0].extract()
            #print("-------------"+c_name+"-----------------")
            target = "https://play.google.com"+url.xpath('@href')[0].extract()
            c_name = "popular"
            yield  scrapy.Request(
                target + "?hl=en",
                meta ={"cat":c_name},
                callback = self.parse_data
            )
            yield  scrapy.Request(
                target + "&hl=en",
                meta ={"cat":c_name},
                callback = self.parse_data
            )

        return

        for url in response.xpath('//div[@class="dropdown-submenu"]/ul/li[@class="action-dropdown-outer-list-item list-item"]'):
            for cat in url.xpath('ul/li[@class="child-submenu-link-wrapper"]/a[@class="child-submenu-link"]'):
                cat_name = cat.xpath('text()')[0].extract()
                with open('cat_log','a+') as f:
                    f.write("category name:"+cat_name+'\n')
                cat_url = "https://play.google.com" + cat.xpath('@href')[0].extract()

                yield scrapy.Request(cat_url+"?hl=en",meta={'cat':cat_name}, callback=self.parse_cat)

    def parse_cat(self, response):

            for url in response.xpath('//a[@class="see-more play-button small id-track-click apps id-responsive-see-more"]'):
                target = "https://play.google.com"+url.xpath("@href")[0].extract()
                yield  scrapy.Request(
                    target + "?hl=en",
                    meta ={"cat":response.meta["cat"]},
                    callback = self.parse_data
                )
                yield  scrapy.Request(
                    target + "&hl=en",
                    meta ={"cat":response.meta["cat"]},
                    callback = self.parse_data
                )

    def parse_data(self, response):
        if response.status == 404:
            return

        for app_obj in response.xpath('//div[@class="card no-rationale square-cover apps small"]/div[@class="card-content id-track-click id-track-impression"]'):
            link = app_obj.xpath('div[@class="details"]/a[@class="title"]/@href')[0].extract()
            #print "LINK:", link

            yield scrapy.Request(
                "https://play.google.com"+link+"&hl=en",
                meta = {"cat":response.meta["cat"]},
                callback = self.parse_app
            )


    def parse_app(self, response):

        more = response.xpath('//a[@class="title-link id-track-click"]/@href')[0].extract()
        #print "MORE:",more
        description = response.xpath('//div[@class="show-more-content text-body"]/div[@jsname="C4s9Ed"]/text()')[0].extract()
        description = "".join(description)
        name = response.xpath('//div[@class="id-app-title"]/text()')[0].extract().strip()
        yield scrapy.Request(
            "https://play.google.com"+more+"&hl=en",
            meta={"name":name, "d":description, "cat":response.meta["cat"]},
            callback=self.parse_sim
        )

    def parse_sim(self, response):

        myItem = GoogleplayItem()
        myItem['app_title'] = response.meta["name"]
        myItem['category'] = response.meta["cat"]
        myItem['description'] = response.meta["d"]
        sim_dict = []
        try:
            for app_obj in response.xpath('//div[@class="card no-rationale square-cover apps small"]/div[@class="card-content id-track-click id-track-impression"]'):
                title = app_obj.xpath('div[@class="details"]/a[@class="title"]/text()')[0].extract()
            #description = app_obj.xpath('div[@class="details"]/div[@class="description"]/text()')[0].extract()
            #description = "".join(description)
            #print "title=",title.strip()," description=",description.strip()
            #myItem['title'] = title.strip()
            #myItem['description'] = description.strip()
            #sim_dict[title.strip()] = description.strip()
                sim_dict.append(title.strip())

        except:
            pass

        myItem['similar_apps'] = sim_dict

        yield myItem

