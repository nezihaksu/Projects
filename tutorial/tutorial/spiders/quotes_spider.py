import scrapy
from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
import sys
sys.path.append('../')
from items import *
from pipelines import *
from bs4 import BeautifulSoup as bs

class QuotesSpider(CrawlSpider):
    
    name = "red"

    allowed_domain = ['reddit.com']

    start_urls = ['https://www.reddit.com/subreddits/leaderboard/'
    ]

    rules = [Rule(LinkExtractor(allow = r"/r/.*/"),callback= 'sub_parser')]

    reddit_pipeline = TutorialPipeline()

    leaderboards = LeaderItem()
    

    def sub_parser(self,response):


        names = response.css('h1._2yYPPW47QxD4lFQTKpfpLQ').getall()
        
        self.leaderboards['leaderboard_names'] = [bs(name).get_text() for name in names]

        member_class = response.css('div.nEdqRRzLEN43xauwtgTmj')

        for members in member_class:
            
            members = members.css('div._3XFx6CfPlg-4Usgxm0gK8R').getall()

            for member in members:

                for letter in bs(member).get_text():
    
                    if letter == 'k':
    
                        self.leaderboards['members'] = [bs(member).get_text().replace(letter,'000')
                        for member in members]
    
                    elif letter == 'm':
    
                        self.leaderboards['members'] = [bs(member).get_text().replace(letter,'000000')
                        for member in members]
        
        online_class = response.css('div._3_HlHJ56dAfStT19Jgl1bF')

        for onlines in online_class: 
            
            onlines = onlines.css('div._3XFx6CfPlg-4Usgxm0gK8R').getall()

            for online in onlines:
    
                for letter in bs(online).get_text():
    
                    if letter == 'k':
    
                        self.leaderboards['onlines']  = [bs(online).get_text().replace(letter,'000') 
                        for online in onlines]
    
                    elif letter == 'm':
    
                        self.leaderboards['onlines'] = [bs(online).get_text().replace(letter,'000000') 
                        for online in onlines]
            
        
            topics = response.css('h3._eYtD2XCVieq6emjKBH3m').getall()
    
            self.leaderboards['topics'] = [bs(topic).get_text() for topic in topics]



        processed_leaderboards = self.reddit_pipeline.process_item(self.leaderboards,QuotesSpider)