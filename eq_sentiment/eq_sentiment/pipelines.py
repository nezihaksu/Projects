# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import pymysql 


class EqSentimentPipeline(object):

	def __init__(self):

		self.open_connection()
		self.create_table()


	def open_connection(self):
		

		self.con = pymysql.connect(

		host = 'localhost',
		user = 'root',
		password = 'lemansser',
		db = 'eq_sentiment',
		charset = 'utf8mb4')

		self.curr = self.con.cursor()

	def create_table(self):

		self.curr.execute('''CREATE TABLE sentiment_magnitude(

		comments TEXT(2000),
		classified_intensity CHAR(20),
		magnitude  VARCHAR(255))''')

	def store_db(self,item):

		insert_query = '''INSERT INTO sentiment_magnitude (comments,classified_intensity,magnitude) VALUES ("%s","%s","%s");'''

		values = ((item['comments'],item['classified_intensity'],
			item['magnitude']))

		self.curr.execute(insert_query,values)
		
		self.con.commit()
		 
	def process_item(self, item, spider):
        
		self.store_db(item)	

		return item
