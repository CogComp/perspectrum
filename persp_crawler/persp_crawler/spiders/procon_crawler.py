import scrapy
from urllib.parse import urlparse
import json

class ProconCrawler(scrapy.Spider):
    name = 'procon'
    start_urls = [
        "https://www.procon.org/"
    ]

    custom_settings = {
        'DOWNLOAD_DELAY': 1,
    }

    def parse(self, response):
        base_url = response.url


        # Get the number of pages from the last page button
        url_list1 = response.css(".newblue-even-line").css("a::attr(href)").extract()
        url_list2 = response.css(".newblue-odd-line").css("a::attr(href)").extract()

        url_list = url_list1 + url_list2

        for url in url_list:
            yield response.follow(url, callback=self.parse_page)

    def parse_page(self, response):
        # First determine type of the pages
        _url = response.url
        if ('headline' in _url) or ('election' in _url):
            yield self.parse_headline_page(response)
        else:
            yield self.parse_normal_page(response)

    def parse_headline_page(self, response):
        return {
            "url": response.url,
            "claim_title": "",
            "topic": [],
            "perspectives_for_count": 0,
            "perspectives_for": [],
            "perspectives_against_count": 0,
            "perspectives_against": [],
        }

    def parse_normal_page(self, response):
        claim_title = response.css(".main-question::text").extract()[0]

        pro_persps = response.css(".newblue-pro-quote-box")
        con_persps = response.css(".newblue-con-quote-box")

        # Positive Perspectives
        perspectives_for = []
        for p in pro_persps:
            persp_title = p.css(".newblue-arguments-bolded-intro::text").extract()[0]
            evidence = p.xpath("./text()").extract()
            evidence = [p for p in evidence if len(p) > 5]  # Filter out stuff like \n or spaces
            persp = {
                "perspective_title": persp_title,
                "argument": {
                    "description": evidence,
                    "reference": []
                },
                "counter_argument": []
            }
            perspectives_for.append(persp)

        # Negative Perspectives
        perspectives_against = []
        for p in con_persps:
            persp_title = p.css(".newblue-arguments-bolded-intro::text").extract()[0]
            evidence = p.xpath("./text()").extract()
            evidence = [p for p in evidence if len(p) > 5]  # Filter out stuff like \n or spaces

            persp = {
                "perspective_title": persp_title,
                "argument": {
                    "description": evidence,
                    "reference": []
                },
                "counter_argument": []
            }
            perspectives_against.append(persp)

        # extract topic from page link
        return {
            "url": response.url,
            "claim_title": claim_title,
            "topic": [],
            "perspectives_for_count": len(perspectives_for),
            "perspectives_for": perspectives_for,
            "perspectives_against_count": len(perspectives_against),
            "perspectives_against": perspectives_against,
        }