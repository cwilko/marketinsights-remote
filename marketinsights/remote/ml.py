import pandas
import pytz
import json
import time
import hashlib
import urllib
import dateutil.parser as parser
import marketinsights.utils.http as http


class MarketInsights():

    def __init__(self, credentials_store, secret="MIOapi_cred"):
        credentials = credentials_store.getSecrets(secret)
        self.credentials = credentials

    def put_dataset(self, data, dataset_desc, market, debug=False):
        dataset = Dataset.csvtojson(data, dataset_desc, market)
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'content-type': 'application/json'
        }
        url = "".join([self.credentials["mi-api-endpoint"], "/v1/datasets"])
        return http.put(url=url, headers=headers, data=dataset, debug=debug)

    def get_dataset(self, dataset_desc, market, debug=False):
        return self.get_dataset_by_id(Dataset.generateId(dataset_desc, market), debug)

    def get_dataset_by_id(self, dataset_id, debug=False):
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'accept': 'application/json'
        }

        # TODO eliminate id, no need.
        query = {
            'where': {
                'id': dataset_id,
            }
        }
        url = "".join([self.credentials["mi-api-endpoint"], "/v1/datasets?filter=", json.dumps(query)])
        resp = http.get(url=url, headers=headers, debug=debug)
        dataset = resp[0]
        return [Dataset.jsontocsv(dataset), dataset["dataset_desc"]]

    # TODO  (without getting data too)
    def get_dataset_desc(self):
        pass

    def put_predictions(self, data, market, modelId, throttle=10, sleep=2, debug=False, update=False):

        # Throttle API calls (throttle = Number of calls/sec)
        if (throttle is not None):
            i = 0
            for j in range(throttle, len(data) + throttle, throttle):
                print("".join(["Sending chunk ", str(j // throttle), " of ", str((len(data) // throttle) + 1)]))
                res = self.put_predictions(data[i:j], market, modelId, throttle=None, debug=debug, update=update)
                if ("error" in res):
                    return res
                time.sleep(sleep)
                i = j
            return {"success": True}
        else:

            # POST Prediction object to API
            data = Predictions.csvtojson(data, market, modelId)
            headers = {
                'X-IBM-Client-Id': self.credentials["clientId"],
                'X-IBM-Client-Secret': self.credentials["clientSecret"],
                'content-type': 'application/json'
            }

            if (update):
                url = "".join([self.credentials["mi-api-endpoint"], "/v1/predictions"])
                for prediction in data:
                    resp = http.put(url=url, headers=headers, data=json.dumps(prediction), debug=debug)

            else:
                url = "".join([self.credentials["mi-api-endpoint"], "/v1/predictions"])
                resp = http.post(url=url, headers=headers, data=json.dumps(data), debug=debug)

            return resp

    # TODO : Deprecated
    def get_predictions(self, market, modelId, start=None, end=None, debug=False):
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'accept': 'application/json'
        }

        query = Predictions.getQuery(market, modelId, start, end)

        url = "".join([self.credentials["mi-api-endpoint"], "/v1/predictions?filter=", json.dumps(query)])
        resp = http.get(url=url, headers=headers, debug=debug)
        return Predictions.jsontocsv(resp)

    def delete_predictions(self, market, modelId, start=None, end=None, debug=False):

        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'content-type': 'application/json'
        }

        data = Predictions.csvtojson(self.get_predictions(market, modelId, start, end, debug), market, modelId)
        if (len(data) > 0):
            for prediction in data:
                url = "".join([self.credentials["mi-api-endpoint"], "/v1/predictions/", prediction["id"]])
                resp = http.delete(url=url, headers=headers, debug=debug)
                if debug:
                    print(prediction)
            return resp
        else:
            return []

    def put_model(self, data, debug=False):
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'content-type': 'application/json'
        }
        url = "".join([self.credentials["mi-api-endpoint"], "/v1/models"])
        return http.put(url=url, headers=headers, data=json.dumps(data), debug=debug)

    def get_model(self, modelId, debug=False):
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'accept': 'application/json'
        }
        url = "".join([self.credentials["mi-api-endpoint"], "/v1/models/", modelId])
        return http.get(url=url, headers=headers, debug=debug)

    def put_training_run(self, data, debug=False):
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'content-type': 'application/json'
        }
        url = "".join([self.credentials["mi-api-endpoint"], "/v1/training_runs"])
        return http.put(url=url, headers=headers, data=json.dumps(data), debug=debug)

    def get_training_run(self, training_run_id, debug=False):
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'accept': 'application/json'
        }
        url = "".join([self.credentials["mi-api-endpoint"], "/v1/training_runs/", training_run_id])
        return http.get(url=url, headers=headers, debug=debug)

    def get_score(self, data, training_run_id, debug=False):
        featureSet = Dataset.csvtojson(data, {}, "Score Data", createId=False)
        headers = {
            'X-IBM-Client-Id': self.credentials["clientId"],
            'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'content-type': 'application/json',
            'accept': 'application/json'

        }
        # TODO : Move to another endpoint
        url = "".join([self.credentials["mi-ml-endpoint"], "/v1/predict/", training_run_id])
        resp = http.post(url=url, headers=headers, data=featureSet, debug=debug)
        if debug:
            print(featureSet)
            print(url)
        return Dataset.jsontocsv(resp)


class Dataset:

    @staticmethod
    def csvtojson(csv, dataset_desc, market, createId=True):
        obj = {}
        if (createId):
            obj["id"] = Dataset.generateId(dataset_desc, market)
        obj["dataset_desc"] = dataset_desc
        obj["market"] = market
        obj["data"] = csv.values.tolist()
        obj["tz"] = csv.index.tz.zone
        obj["index"] = [date.isoformat() for date in csv.index.tz_localize(None)]  # Remove locale
        return json.dumps(obj)

    @staticmethod
    def jsontocsv(jsonObj):
        return pandas.DataFrame(jsonObj["data"], index=pandas.DatetimeIndex(jsonObj["index"], name="Date_Time", tz=pytz.timezone(jsonObj["tz"])))

    @staticmethod
    def generateId(dataset_desc, market):
        return hashlib.md5("".join([market, json.dumps(dataset_desc["pipeline"], sort_keys=True), str(dataset_desc["features"]), str(dataset_desc["labels"])]).encode('utf-8')).hexdigest()


class Predictions:

    @staticmethod
    def csvtojson(data, market, modelId):
        data.index = data.index.tz_localize(None)
        obj = [{
            "id": hashlib.md5("".join([modelId, "_", market, "_", i.isoformat()]).encode('utf-8')).hexdigest(),
            "market":market,
            "model_id":modelId,
            "timestamp":i.isoformat(),
            "data":data.loc[i].values.tolist()} for i in data.index]
        return obj

    @staticmethod
    def jsontocsv(jsonObj):
        idx = pandas.DatetimeIndex([prediction["timestamp"] for prediction in jsonObj])
        return pandas.DataFrame([prediction["data"] for prediction in jsonObj], idx).sort_index()

    @staticmethod
    def getQuery(market, modelId, start, end):
        query = {
            'where': {
                'market': market,
                'model_id': modelId
            }
        }

        if (start is not None and end is not None):
            query["where"]["timestamp"] = {'between': [parser.parse(start).isoformat(), parser.parse(end).isoformat()]}
        else:
            if (start is not None):
                query["where"]["timestamp"] = {'gte': parser.parse(start).isoformat()}

            if (end is not None):
                query["where"]["timestamp"] = {'lte': parser.parse(end).isoformat()}

        return query
