import json
import quantutils.dataset.pipeline as ppl


class MarketInsightsAssembly:

    def __init__(self, mi, fun):
        # TODO: Create the instances of MI and Functions in here
        self.mi = mi
        self.fun = fun

    def get_predictions_with_dataset(self, dataset, training_run_id, debug=False):

        # Send the dataset features to the model and retrieve the scores (predictions)
        return self.mi.get_score(dataset, training_run_id, debug)

    def get_predictions_with_dataset_id(self, dataset_id, training_run_id, start=None, end=None, debug=False):

        # Get the dataset from storage, crop and strip out labels
        dataset, _ = self.mi.get_dataset_by_id(dataset_id, debug)
        dataset = dataset[start:end].iloc[:, :-1]

        if debug:
            print(dataset)

        if dataset.empty:  # No predictions in this time range
            return None

        return self.get_predictions_with_dataset(dataset, training_run_id, debug)

    def get_predictions_with_raw_data(self, data, training_id, debug=False):

        training_run = self.mi.get_training_run(training_id)
        if debug:
            print("Training run : " + str(training_run))

        dataset_id = training_run["datasets"][0]
        dataset_desc = self.mi.get_dataset_by_id(dataset_id)[1]
        pipeline = dataset_desc["pipeline"]
        if debug:
            print("Pipeline info : " + str(pipeline))

        # Generate a dataset from the raw data through the given pipeline
        inputData = {
            "data": json.loads(data.to_json(orient='split', date_format="iso")),
            "dataset": pipeline["pipeline_desc"]
        }

        if debug:
            print("Request to pipeline : " + str(inputData))

        dataset = self.fun.call_function(pipeline["id"], inputData, debug)

        if debug:
            print("Pipeline response : " + str(dataset))

        dataset.index.names = ["Date_Time"]  # Workaround for lack of index name in marshalling
        csvData = ppl.localize(dataset, "UTC", pipeline["pipeline_desc"]["timezone"])  # Marshalling turns dates to UTC
        csvData = dataset.iloc[:, :-dataset_desc["labels"]]

        if debug:
            print("Sending feature vector : " + str(csvData))
        return self.get_predictions_with_dataset(csvData, training_id, debug)

    # Local client
    def get_local_predictions_with_dataset_id(self, mi_client, dataset_id, training_run_id, start=None, end=None, debug=False):

        # Get the dataset from storage, crop and strip out labels
        dataset, _ = self.mi.get_dataset_by_id(dataset_id)
        dataset = dataset[start:end].iloc[:, :-1]

        if debug:
            print(dataset)

        if dataset.empty:  # No predictions in this time range
            return None

        obj = {}
        obj["data"] = dataset.values.tolist()
        obj["tz"] = dataset.index.tz.zone
        obj["index"] = [date.isoformat() for date in dataset.index.tz_localize(None)]

        return mi_client.score(training_run_id, obj)
