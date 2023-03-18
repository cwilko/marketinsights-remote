import pandas
from urllib.parse import quote
import marketinsights.utils.http as http


class MIDataStoreRemote:

    def __init__(self, location):
        self.mdsRemote = PriceStoreAPI(location)

    def aggregate(self, table_id, sources, start="1979-01-01", end="2050-01-01", debug=False):
        results = self.mdsRemote.aggregate(table_id, sources, start, end, debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return pandas.read_json(results["body"], orient="split", dtype=False).set_index(["Date_Time", "ID"])
        return pandas.DataFrame()

    def get(self, table_id, debug=False):
        results = self.mdsRemote.get(table_id, debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return pandas.read_json(results["body"], orient="split", dtype=False).set_index(["Date_Time", "ID"])
        return pandas.DataFrame()

    def append(self, table_id, data, update=False, debug=False):
        if update:
            return self.mdsRemote.put(table_id, data.reset_index().to_json(orient='split', date_format="iso"), debug)
        else:
            return self.mdsRemote.post(table_id, data.reset_index().to_json(orient='split', date_format="iso"), debug)

    def delete(self, table_id, debug=False):
        return self.mdsRemote.delete(table_id, debug)

    def getKeys(self, debug=False):
        results = self.mdsRemote.getKeys(debug)
        if (results["rc"] == "success" and results["body"] is not None):
            return results["body"]
        return None


class PriceStoreAPI:

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def get(self, table_id, debug=False):
        headers = {
            'accept': 'application/json'
        }
        url = "".join([self.endpoint, "/prices/datasource/", table_id])
        return http.get(url=url, headers=headers, debug=debug)

    def aggregate(self, table_id, sources, start, end, debug=False):
        headers = {
            'accept': 'application/json'
        }
        url = "".join([self.endpoint, "/prices/aggregate/", table_id, "?start=", start, "&end=", end, "&sources=", "&sources=".join([quote(source) for source in sources])])
        return http.get(url=url, headers=headers, debug=debug)

    def put(self, table_id, data, debug=False):
        headers = {
            'Content-Type': 'application/json',
        }
        url = "".join([self.endpoint, "/prices/datasource/", table_id])
        return http.put(url=url, headers=headers, data=data, debug=debug)

    def post(self, table_id, data, debug=False):
        headers = {
            'Content-Type': 'application/json'
        }
        url = "".join([self.endpoint, "/prices/datasource/", table_id])
        return http.post(url=url, headers=headers, data=data, debug=debug)

    def delete(self, table_id, debug=False):
        headers = {
            'accept': 'application/json'
        }
        url = "".join([self.endpoint, "/prices/datasource/", table_id])
        return http.delete(url=url, headers=headers, debug=debug)

    def getKeys(self, debug=False):
        headers = {
            'accept': 'application/json'
        }
        url = "".join([self.endpoint, "/prices/keys"])
        return http.get(url=url, headers=headers, debug=debug)
