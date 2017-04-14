# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
import sqlite3
import os

fileName = 'googleplay.json'
db = 'googleplay.db'

class GoogleplayPipeline(object):
    def __init__(self):
        #json
        #with open(fileName, 'w') as f:
        #     f.write('[\n')
        pass

    def open_spider(self, spider):
        #sqlite
        #self.con = sqlite3.connect(db)
        #self.cur = self.con.cursor()
        #self.cur.execute('DROP TABLE IF EXISTS googleplay')
        #self.cur.execute('create table if not exists googleplay(table_title varchar(20),title varchar(50), title_URL varchar(100), imgURL varchar(100), description varchar(250), autor varchar(30),      autor_URL varchar(100), star_rates varchar(20), price varchar(10))')
        pass

    def close_spider(self, spider):

        for fname in os.listdir("./"):
            if fname.endswith(".json"):
                with open(fname, 'r') as f:
                    content = f.read()
                with open(fname, 'w') as f:
                    f.write( content[:-1] + "\n]" )


    def process_item(self, item, spider):

        #json
        line = json.dumps(dict(item), ensure_ascii = False, encoding = 'utf8', indent = 4 ) + ','
        cate = dict(item)['category']

        if not os.path.exists(cate+".json"):
            with open(cate+".json", 'w') as f:
                f.write("[\n")

        with open(cate+".json", 'a') as f:
             f.write( line.encode('utf8') )

        return item

