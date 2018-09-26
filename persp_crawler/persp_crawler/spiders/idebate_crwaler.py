import scrapy
from urllib.parse import urlparse
import json

class IdebateCrawler(scrapy.Spider):
    name = 'idebate'
    start_urls = [
        "https://idebate.org/debatabase/culture",
        "https://idebate.org/debatabase/economy",
        "https://idebate.org/debatabase/education",
        "https://idebate.org/debatabase/environment",
        "https://idebate.org/debatabase/free-speech-debate",
        "https://idebate.org/debatabase/health",
        "https://idebate.org/debatabase/international",
        "https://idebate.org/debatabase/law",
        "https://idebate.org/debatabase/philosophy",
        "https://idebate.org/debatabase/politics",
        "https://idebate.org/debatabase/religion",
        "https://idebate.org/debatabase/science",
        "https://idebate.org/debatabase/society",
        "https://idebate.org/debatabase/sport",
        "https://idebate.org/debatabase/digital-freedoms"
    ]

    custom_settings = {
        'DOWNLOAD_DELAY': 1,
    }

    def parse(self, response):
        base_url = response.url
        listing_urls = [base_url]

        print("Processing:\t{}".format(base_url))
        # Get the number of pages from the last page button
        _num_pages = response.css("li.page-numbers.last").re('\?page=(\d+)')

        num_pages = -1
        if _num_pages:
            num_pages = int(_num_pages[0])

        for i in range(num_pages):
            listing_urls.append(base_url + "?page=" + str(i))

        for url in listing_urls:
            yield response.follow(url, callback=self.parse_listing_page)

    def parse_listing_page(self, response):
        pages = response.css('div.debatabase-theme-listing-title a::attr(href)').extract()
        for page_url in pages:
            yield response.follow(page_url, callback=self.parse_page)

    def parse_page(self, response):
        claim_title = response.css("div.debatabase-title::text").extract()[0]
        _description_paragraphs = response.css("div.debatabase-description p")

        # list of paragraphs in claim description
        description = []
        for p in _description_paragraphs:
            description.append(" ".join(p.css("*::text").extract()))

        # keep references links and filter out links that doesn't start with 'http'
        refs_in_descriptions = response.css("div.debatabase-description a::attr(href)").extract()
        refs_in_descriptions = [l for l in refs_in_descriptions if l.startswith("http")]

        # Now extract all "for" perspectives
        perspectives_for = []
        _persps = response.css("div#debatabase-points-1 div.entity-field-collection-item")

        for p in _persps:
            title = p.css("div.field-name-field-title-point-for div.field-item::text").extract()[0]
            point_paragraphs = []
            for pp in p.css("div.field-name-field-point-point-for p"):
                point_paragraphs.append(pp.css("*::text").extract())

            point_refs = p.css("div.field-name-field-point-point-for a::attr(href)").extract()
            point_refs = [l for l in point_refs if l.startswith("http")]

            counterpoint_paragraphs = []
            for cpp in p.css("div.field-name-field-counterpoint-point-for"):
                counterpoint_paragraphs.append(cpp.css("*::text").extract())

            counterpoint_refs = p.css("div.field-name-field-counterpoint-point-for a::attr(href)").extract()
            counterpoint_refs = [l for l in counterpoint_refs if l.startswith("http")]

            persp = {
                "title": title,
                "point": {
                    "description" : point_paragraphs,
                    "reference": point_refs
                },
                "counterpoint": {
                    "description": counterpoint_paragraphs,
                    "reference": counterpoint_refs
                }
            }

            perspectives_for.append(persp)

        # Do the same for "against" perspectives
        perspectives_against = []
        _persps = response.css("div#debatabase-points-2 div.entity-field-collection-item")

        for p in _persps:
            title = p.css("div.field-name-field-title div.field-item::text").extract()[0]
            point_paragraphs = []
            for pp in p.css("div.field-name-field-point p"):
                point_paragraphs.append(pp.css("*::text").extract())

            point_refs = p.css("div.field-name-field-point a::attr(href)").extract()
            point_refs = [l for l in point_refs if l.startswith("http")]

            counterpoint_paragraphs = []
            for cpp in p.css("div.field-name-field-counterpoint"):
                counterpoint_paragraphs.append(cpp.css("*::text").extract())

            counterpoint_refs = p.css("div.field-name-field-counterpoint a::attr(href)").extract()
            counterpoint_refs = [l for l in counterpoint_refs if l.startswith("http")]

            persp = {
                "perspective_title": title,
                "perspective_point": {
                    "description" : point_paragraphs,
                    "reference": point_refs
                },
                "perspective_counterpoint": {
                    "description": counterpoint_paragraphs,
                    "reference": counterpoint_refs
                }
            }

            perspectives_against.append(persp)

        # extract topic from page link
        topic = urlparse(response.url).path.split("/")[2]
        yield {
            "claim_title": claim_title,
            "topic": topic,
            "perspectives_for_count": len(perspectives_for),
            "perspectives_for": perspectives_for,
            "perspectives_against_count": len(perspectives_against),
            "perspectives_against": perspectives_against,
        }
