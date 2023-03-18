import marketinsights.utils.http as http
import json


class TradeFramework:

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def createEnvironment(self, name, tz, debug=False):
        headers = {
            'accept': 'application/json'
        }
        url = "".join([self.endpoint, "/environments?name=", name, "&tz=", tz])
        return http.post(url=url, headers=headers, data={}, debug=debug)

    def createPortfolio(self, env_uuid, p_uuid, name, optimizer, opts={}, debug=False):
        headers = {
            'Content-Type': 'application/json'
        }
        url = "".join([self.endpoint, "/environments/", env_uuid, "/portfolios/", p_uuid, "/portfolios?name=", name, "&optimizer=", optimizer])
        return http.post(url=url, headers=headers, data=json.dumps({"options": opts}), debug=debug)

    def createModel(self, env_uuid, p_uuid, name, modelType, opts={}, debug=False):
        headers = {
            'Content-Type': 'application/json'
        }
        url = "".join([self.endpoint, "/environments/", env_uuid, "/portfolios/", p_uuid, "/models?name=", name, "&type=", modelType])
        return http.post(url=url, headers=headers, data=json.dumps({"options": opts}), debug=debug)

    def addStoredAsset(self, env_uuid, p_uuid, name, debug=False):
        headers = {
            'Content-Type': 'application/json'
        }
        url = "".join([self.endpoint, "/environments/", env_uuid, "/portfolios/", p_uuid, "/assets?name=", name])
        return http.post(url=url, headers=headers, data={}, debug=debug)

    def appendAsset(self, env_uuid, market, prices, debug=False):
        headers = {
            'Content-Type': 'application/json'
        }
        url = "".join([self.endpoint, "/environments/", env_uuid, "/assets?market=", market])
        return http.post(url=url, headers=headers, data=json.dumps(prices), debug=debug)

    def refresh(self, env_uuid, debug=False):
        headers = {
            'Content-Type': 'application/json'
        }
        url = "".join([self.endpoint, "/environments/", env_uuid])
        return http.post(url=url, headers=headers, data={}, debug=debug)

    def getSignal(self, env_uuid, capital, debug=False):
        headers = {
            'accept': 'application/json'
        }
        url = "".join([self.endpoint, "/environments/", env_uuid, "/signals?capital=", str(capital)])
        return http.get(url=url, headers=headers, debug=debug)

    def createPredictions(self, env_uuid, prices, market, capital, debug=False):
        headers = {
            'Content-Type': 'application/json'
        }
        url = "".join([self.endpoint, "/environments/", env_uuid, "/predictions?capital=", str(capital), "&market=", market])
        return http.post(url=url, headers=headers, data=json.dumps({"prices": prices}), debug=debug)
