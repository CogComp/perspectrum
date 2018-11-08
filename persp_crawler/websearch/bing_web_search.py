import http.client, urllib.parse, json
from time import sleep

# Replace the subscriptionKey string value with your valid subscription key.
subscriptionKey = "62c2421f14994e7ebf4400d3e130a3bc"

# Verify the endpoint URI.  At this writing, only one endpoint is used for Bing
# search APIs.  In the future, regional endpoints may be available.  If you
# encounter unexpected authorization errors, double-check this value against
# the endpoint for your Bing Web search instance in your Azure dashboard.
host = "api.cognitive.microsoft.com"
path = "/bing/v7.0/search"


class BingWebSearch:

    def __init__(self, secret_key, timeout):
        """
        :param secret_key: Azure subscription key
        :param timeout: timeout in seconds after each query (M$ limits queries per second according to pricing tier)
        """

        self.secret_key = secret_key
        self.timeout = timeout


    def search(self, search_term):
        """
        Performs a Bing Web search and returns the results.
        :param search_term: 
        :return: decoded http response
        """
        sleep(self.timeout)  # TODO: very crude but works...
        _headers = {'Ocp-Apim-Subscription-Key': self.secret_key}
        conn = http.client.HTTPSConnection(host)
        query = urllib.parse.quote(search_term)
        conn.request("GET", path + "?q=" + query, headers=_headers)
        response = conn.getresponse()
        _headers = [k + ": " + v for (k, v) in response.getheaders() if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")]
        return _headers, response.read().decode("utf8")


if __name__ == '__main__':

    engine = BingWebSearch(subscriptionKey, 1)
    term = "Should abortion be legal"

    if len(subscriptionKey) == 32:

        print('Searching the Web for: ', term)

        headers, result = engine.search(term)
        print("\nRelevant HTTP Headers:\n")
        print("\n".join(headers))
        print("\nJSON Response:\n")
        print(json.dumps(json.loads(result), indent=4))

    else:

        print("Invalid Bing Search API subscription key!")
        print("Please paste yours into the source code.")