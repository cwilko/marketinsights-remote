import pandas
import pytz
import json
import time
import hashlib
import urllib
import dateutil.parser as parser
import quantutils.dataset.pipeline as ppl
import marketinsights.utils.http as http
from marketinsights.remote.models import MIModelServer
from marketinsights.remote.ibmcloud import CloudFunctions
from marketinsights.utils.auth import CredentialsStore


class MIAssembly():

    def __init__(self, modelSvr=None, credentials_store=None, secret="marketinsights-local-cred"):

        if not credentials_store:
            credentials_store = CredentialsStore()

        self.fun = CloudFunctions(credentials_store)
        self.credentials = credentials_store.getSecrets(secret)
        self.modelSvr = modelSvr

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
        return self.get_dataset_by_id(Dataset.generateMarketId(dataset_desc, market), debug)

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

    def get_predictions_with_dataset(self, dataset, training_run_id, debug=False):

        if not self.modelSvr:
            raise Exception("No Model Server configured")

        # Send the dataset features to the model and retrieve the scores (predictions)
        return self.modelSvr.getPredictions(dataset, training_run_id, debug=debug)

    def get_predictions_with_dataset_id(self, dataset_id, training_run_id, start=None, end=None, debug=False):

        # Get the dataset from storage, crop and strip out labels
        dataset, _ = self.get_dataset_by_id(dataset_id, debug)
        dataset = dataset[start:end].iloc[:, :-1]

        if debug:
            print(dataset)

        if dataset.empty:  # No predictions in this time range
            return None

        return self.get_predictions_with_dataset(dataset, training_run_id, debug)

    def get_predictions_with_raw_data(self, data, training_id, debug=False):

        training_run = self.get_training_run(training_id)
        if debug:
            print("Training run : " + str(training_run))

        dataset_id = training_run["datasets"][0]
        dataset_desc = self.get_dataset_by_id(dataset_id)[1]
        pipeline = dataset_desc["pipeline"]
        if debug:
            print("Pipeline info : " + str(pipeline))

        # Generate a dataset from the raw data through the given pipeline
        dataset = Dataset().executePipeline(pipeline["id"], data, pipeline["pipeline_desc"], debug)

        if dataset.empty:
            return dataset

        dataset = dataset.iloc[:, :-dataset_desc["labels"]]
        if debug:
            print("Sending feature vector : " + str(dataset))

        return self.get_predictions_with_dataset(dataset, training_id, debug)

    @staticmethod
    def generateMarketId(dataset_desc, market):
        return hashlib.md5("".join([market, json.dumps(dataset_desc["pipeline"], sort_keys=True), str(dataset_desc["features"]), str(dataset_desc["labels"])]).encode('utf-8')).hexdigest()

    @staticmethod
    def generateTrainingId(dataset_desc, model_id, name=None):
        training_id = str(hashlib.md5(f'{str(dataset_desc)}{str(model_id)}'.encode('utf-8')).hexdigest())
        if name:
            training_id = f"{name}-{training_id}"
        return training_id


class Dataset:

    def __init__(self, fun=None, credentials_store=None):

        if not credentials_store:
            credentials_store = CredentialsStore()
        if not fun:
            fun = CloudFunctions(credentials_store)
        self.fun = fun

    def executePipeline(self, pipeline, data, config, debug=False):

        if debug:
            print("Request to pipeline : " + str(data))

        data = {
            "data": json.loads(data.to_json(orient='split', date_format="iso")),
            "dataset": config
        }

        response = self.fun.call_function(pipeline, data, debug)

        if debug:
            print("Pipeline response : " + str(response))

        if response:
            dataset = pandas.read_json(json.dumps(response), orient='split', dtype=False)
            if not dataset.empty:  # Check for empty dataset
                dataset.index.names = ["Date_Time"]  # Workaround for lack of index name in marshalling
                dataset = ppl.localize(dataset, "UTC", config["timezone"])  # Marshalling turns dates to UTC
            return dataset
        else:
            raise Exception("No response received from pipeline execution")

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
