import scrapy
from urllib.parse import urlparse
import json

class DebatewiseCrawler(scrapy.Spider):
    name = 'debatewise'
    start_urls = [
        "https://debatewise.org"
    ]

    custom_settings = {
        'DOWNLOAD_DELAY': 1,
    }

    def parse(self, response):
        base_url = response.url
        listing_urls = [base_url]

        print("Processing:\t{}".format(base_url))
        # Get the number of pages from the last page button
        _num_pages = response.css("nav .wp-pagenavi .last::attr(href)").re('page/(\d+)/')

        num_pages = -1
        if _num_pages:
            num_pages = int(_num_pages[0])

        for i in range(2, num_pages + 1):
            listing_urls.append(base_url + "/page/" + str(i))

        for url in listing_urls:
            yield response.follow(url, callback=self.parse_listing_page)

    def parse_listing_page(self, response):
        pages = response.css('.hometitle a::attr(href)').extract()
        for page_url in pages:
            yield response.follow(page_url, callback=self.parse_page)

    def parse_page(self, response):
        claim_title = response.css('#wiki-page-wrapper h1::text').extract()[0]

        persps_els = response.css('.pointTitles a')
        persp_titles = persps_els.xpath('./text()').extract()
        persp_ids = persps_els.xpath('./@href').extract()

        perspectives_for = []
        perspectives_against = []
        for i in range(len(persp_ids)):
            persp_title = persp_titles[i]
            persp_id = persp_ids[i]

            is_persp_for = persp_id.startswith("#yes")

            evidence = response.css(persp_id).xpath('.//div[contains(@class, "pointArgument")]//text()').extract()
            evidence = [p.strip() for p in evidence]
            evidence = [p for p in evidence if p]

            persp = {
                "perspective_title": persp_title,
                "argument": {
                    "description": evidence,
                    "reference": []
                },
                "counter_argument": []
            }

            if is_persp_for:
                perspectives_for.append(persp)
            else:
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
