import json
import hashlib
import numpy as np
import pandas as pd
from marketinsights.remote.filesystem import RemoteFS
from marketinsights.remote.grpc import GRPCClient
from marketinsights.utils.auth import CredentialsStore
import marketinsights.utils.http as http

from google.protobuf import json_format
from tensorflow_serving.apis import model_service_pb2_grpc, model_management_pb2
from tensorflow_serving.config import model_server_config_pb2


class MIModelServer:

    def __init__(self, serverConfigPath=None, credentials_store=None, secret="marketinsights-local-cred"):

        if not credentials_store:
            credentials_store = CredentialsStore()

        self.credentials = credentials_store.getSecrets(secret)
        self.remotefs = RemoteFS(credentials_store.getSecrets("scp_cred"))
        self.modelSvr = GRPCClient(credentials_store.getSecrets("model-svr-grpc-secret"))

        self.serverConfigPath = serverConfigPath
        if serverConfigPath:
            self.loadConfig(serverConfigPath)

    def saveConfig(self, filePath):
        with open(filePath, "w") as outfile:
            outfile.write(json.dumps(self.serverConfig, indent=4))

    def loadConfig(self, filePath):
        with open(filePath, 'r') as openfile:
            self.serverConfig = json.load(openfile)
        return self.serverConfig

    def deployModel(self, model, version=1, updateConfig=True):
        model.save(root="/tmp/models", version=version)
        self.remotefs.put(sourcePath=f"/tmp/models/{model.modelName}")

        if updateConfig:
            # Merge model into server config
            self.serverConfig["modelConfigList"]["config"].append(
                {
                    "name": model.modelName,
                    "base_path": f"/models/{model.modelName}",
                    "model_platform": "tensorflow"
                }
            )

            result = {}
            for c in self.serverConfig["modelConfigList"]["config"]:
                result.setdefault(c["name"], {}).update(c)
            self.serverConfig["modelConfigList"]["config"] = list(result.values())

            self.refreshModelServerConfig()

            if self.serverConfigPath:
                self.saveConfig(self.serverConfigPath)

    def refreshModelServerConfig(self):

        # Get gRPC secure connection
        channel = self.modelSvr.getChannel()

        # Parse json server config to proto message
        model_server_config = model_server_config_pb2.ModelServerConfig()
        model_server_config = json_format.Parse(text=json.dumps(self.serverConfig), message=model_server_config)

        # Create a Reload Request
        request = model_management_pb2.ReloadConfigRequest()
        request.config.CopyFrom(model_server_config)

        # Call the service
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        response = stub.HandleReloadConfigRequest(request, 10)

        if response.status.error_code != 0:
            raise Exception(f"Error: {response.status.error_code}, {response.status.error_message}")

    def restoreModel(self, model, modelName=None, version=1):

        if not modelName:
            modelName = model.modelName

        self.remotefs.get(sourceFile=f"{modelName}/{version}", targetPath="/tmp/models/")
        model.restore(f"/tmp/models/{modelName}/{version}")

    def getPredictions(self, data, modelName, version=1, debug=False):
        headers = {
            #'X-IBM-Client-Id': self.credentials["clientId"],
            #'X-IBM-Client-Secret': self.credentials["clientSecret"],
            'content-type': 'application/json',
            'accept': 'application/json'

        }
        payload = Predictions.data_to_json(data)
        url = f'{self.credentials["modelserver-endpoint"]}/v1/models/{modelName}:predict'
        resp = http.post(url=url, headers=headers, data=payload, debug=debug)
        if debug:
            print(payload)
        return Predictions.json_to_data(data, resp)


class Predictions:

    @staticmethod
    def data_to_json(data):
        payload = {
            "signature_name": "predictions",
            "instances": data.values.tolist()
        }
        return json.dumps(payload)

    @staticmethod
    def json_to_data(data, json):
        predictions = np.array(json["predictions"])
        return pd.concat([data, pd.DataFrame(predictions, index=data.index, columns=[f'y_pred{y}' for y in range(predictions.shape[1])])], axis=1)
