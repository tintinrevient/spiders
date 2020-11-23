import scrapy
import string
import json
import urllib

# scrapy startproject spiders
# cd spiders
# scrapy genspider example example.com
# scrapy runspider example.py

class NprSpider(scrapy.Spider):

    name = 'npr'
    allowed_domains = ['www.npg.org.uk']
    start_urls = ['https://www.npg.org.uk/collections/search/sita-z/']

    def parse(self, response):

    	base_url = 'https://www.npg.org.uk/collections/search/sita-z/?'

    	for alphabet in string.ascii_lowercase:
    		# send request
            yield scrapy.Request(base_url + urllib.parse.urlencode({'index': alphabet}), dont_filter = True, callback = self.parse_page)

    def parse_page(self, response):

    	base_url = 'https://www.npg.org.uk'

    	# print response.url
    	print('Processing', response.url)

    	# scrape page info
    	for sitter in response.css('#eventsListing p'):

    		info = sitter.css('::text').extract()
    		
    		yield {
    			'name': info[0],
    			'intro': info[1],
    			'url': base_url + sitter.css('a ::attr(href)').extract_first()
    		}

