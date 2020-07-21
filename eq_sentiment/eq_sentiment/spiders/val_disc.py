import scrapy
from scrapy.spiders import Rule,CrawlSpider
from scrapy.linkextractors import LinkExtractor

from bs4 import BeautifulSoup as bs

import re

import sys
sys.path.append('../')

from items import *
from pipelines import *


class EarthQuakeSpider(CrawlSpider):

	name = 'eq_spider'

	start_urls = ['https://www.volcanodiscovery.com/i-felt-it-reports/earthquakes.html'] 

	rules = [Rule(LinkExtractor(restrict_css = 'a.sl2'),callback = 'eq_parse')]

	sentiment_pipeline = EqSentimentPipeline()

	earthquakes = EqSentimentItem()

	def eq_parse(self,response):

		#Replacing the unwanted characters with empty string.
		def multiple_replace(dict,text):

			regex = re.compile("%s"%"|".join(map(re.escape,dict.keys())))

			return regex.sub(lambda mo:dict[mo.string[mo.start():mo.end()]],text)

		#Catches the phrase.
		magnitude_re = re.compile('Magnitude:*.*')
		depth_re = re.compile('Depth:*.*')
		comments_re = re.compile(r':\w*')
		intensity_re = re.compile(r'(\w+ shaking)')
		epicenter_re = re.compile(r'\d+.\d \w+')
		km_re = re.compile(' km')
		shaking_re = re.compile(' shaking')
		via_colon_dict = {'(via ':'',':':''}
		depth_km_dict = {' km':'','Depth:':''}

		comments = response.xpath('//*[@id="content"]/div/i/text()').getall()
		#Filters through the returned list by xpath and matches according to first character of the list element.
		comments = list(filter(comments_re.match,comments))
		#Unwanted characters are replaced.
		comments = [multiple_replace(via_colon_dict,comment) for comment in comments]
				
		km_from_epicenter = response.xpath('//*[@id="content"]/div/b/text()[1]').re(epicenter_re)
		km_from_epicenter = [re.sub(km_re,'',epicenter) for epicenter in km_from_epicenter]
			
		classified_intensity = response.xpath('//*[@id="content"]/div/text()').re(intensity_re)
		classified_intensity = [re.sub(shaking_re,'',intensity) for intensity in classified_intensity]
			
		magnitude = response.xpath('//div').re(magnitude_re)[-1]
		magnitude = bs(magnitude.replace('Magnitude:',''),features = 'lxml').get_text()

		depth = response.xpath('//div').re(depth_re)[-2]
		depth = bs(multiple_replace(depth_km_dict,depth),features = 'lxml').get_text()
			
		self.earthquakes['comments'] = comments
		self.earthquakes['km_from_epicenter'] = km_from_epicenter
		self.earthquakes['classified_intensity'] = classified_intensity
		self.earthquakes['magnitude'] = magnitude
		self.earthquakes['depth'] = depth

		self.sentiment_pipeline.process_item(self.earthquakes,EarthQuakeSpider)
			



			

		#print(comments,classified_intensity,magnitude,depth,km_from_epicenter)


