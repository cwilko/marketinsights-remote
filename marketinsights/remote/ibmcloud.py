
###################
## Bluemix Setup ##
###################

from io import StringIO, BytesIO
import os
import requests
import json
import pandas as pd
import ibm_boto3
import ibm_botocore
from marketinsights.utils.auth import CredentialsStore


class Token:

    def __init__(self, credentials_store):
        self.iam_cred = credentials_store.getSecrets('iam_cred')

    def getToken(self):
        headers = {'Content-Type': 'application/x-www-form-urlencoded', }
        data = 'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=' + self.iam_cred['apikey']
        resp = requests.post(url=self.iam_cred['endpoint'], headers=headers, data=data)
        return resp.json()


class Metrics:

    def __init__(self, credentials_store):
        self.credentials = credentials_store.getSecrets('metrics_cred')

    def send(self, data):
        headers = {}
        headers['Content-Type'] = 'application/json'
        headers['x-auth-scope-id'] = ''.join(['s-', self.credentials['space_id']])
        headers['x-auth-user-token'] = ''.join(['apikey ', self.credentials['token']])
        resp = requests.post(url=self.credentials['host'], headers=headers, data=json.dumps(data))
        return resp

# Deprecated (see CloudObjectStore)


class ObjectStore:

    def __init__(self, credentials_store):
        credentials = credentials_store.getSecrets('object_storage_cred')
        self.load_obj_storage_token(credentials)

    def load_obj_storage_token(self, obj_storage_cred):

        url = ''.join([obj_storage_cred['auth_url'], '/v3/auth/tokens'])
        data = {'auth': {'identity': {'methods': ['password'],
                                      'password': {'user': {'name': obj_storage_cred['username'], 'domain': {'id': obj_storage_cred['domainId']},
                                                            'password': obj_storage_cred['password']}}}}}
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(url=url, data=json.dumps(data), headers=headers)
        resp_body = resp.json()
        for e1 in resp_body['token']['catalog']:
            if(e1['type'] == 'object-store'):
                for e2 in e1['endpoints']:
                    if(e2['interface'] == 'public'and e2['region'] == 'dallas'):
                        endpoint_url = e2['url']
        token = resp.headers['x-subject-token']

        self.endpoint_url = endpoint_url
        self.token = token

    def put_file(self, container, local_file_name, filename):
        """This functions returns a StringIO object containing
        the file content from Bluemix Object Storage V3."""

        f = open(local_file_name, 'r')
        headers = {'X-Auth-Token': self.token, 'accept': 'application/json'}
        url = "".join([self.endpoint_url, "/", container, "/", filename])
        resp = requests.put(url=url, headers=headers, data=f.read())
        print(resp.text)

    def get_file(self, container, filename):
        """This functions returns a StringIO object containing
        the file content from Bluemix Object Storage."""

        url = "".join([self.endpoint_url, "/", container, "/", filename])
        headers = {'X-Auth-Token': self.token, 'accept': 'application/json'}
        resp = requests.get(url=url, headers=headers)
        return StringIO(resp.text)


class CloudObjectStore:

    def __init__(self, credentials_store):
        credentials = credentials_store.getSecrets('ibm_cos_cred')
        self.cos = self.connect(credentials)

    def connect(self, credentials):
        return ibm_boto3.resource('s3',
                                  ibm_api_key_id=credentials["apikey"],
                                  ibm_service_instance_id=credentials["resource_instance_id"],
                                  ibm_auth_endpoint=credentials["auth_endpoint"],
                                  config=ibm_botocore.client.Config(signature_version='oauth'),
                                  endpoint_url=credentials["service_endpoint"])

    def put(self, bucket, key, local_file_name):
        self.getOrCreateBucket(bucket)
        self.cos.Object(bucket, key).put(Body=open(local_file_name, 'rb'))

    def delete(self, bucket, key):
        self.cos.Object(bucket, key).delete()

    def put_csv(self, bucket, key, csv):
        csv.to_csv('tmp-gz.csv', index=False, compression='gzip')
        self.put(bucket, key, 'tmp-gz.csv')
        os.remove('tmp-gz.csv')

    def get_csv(self, bucket, key):
        if (not self.keyExists(bucket, key)):
            return pd.DataFrame()
        return pd.read_csv(BytesIO(self.cos.Object(bucket, key).get()['Body'].read()), compression='gzip')

    def getOrCreateBucket(self, bucket):
        exists = False
        for cosBucket in self.cos.buckets.all():
            if (bucket == cosBucket.name):
                exists = True

        if (not exists):
            self.cos.create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': 'us-standard'})

    def keyExists(self, bucket, key):
        exists = True
        try:
            self.cos.Object(bucket, key).load()
        except ibm_botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                exists = False
            else:
                # Something else has gone wrong.
                raise
        return exists


class CloudFunctions:

    def __init__(self, credentials_store=None):

        if not credentials_store:
            credentials_store = CredentialsStore()

        self.credentials = credentials_store.getSecrets('functions_cred')
        self.token = Token(credentials_store).getToken()

    def call_function(self, name, args, debug=False):
        headers = {'Content-Type': 'application/json', 'Authorization': "".join(["Bearer ", self.token['access_token']])}
        url = "".join([self.credentials["endpoint"], name, "?blocking=true"])
        resp = requests.post(url=url, data=json.dumps(args), headers=headers)

        if (debug):
            print(resp.text)

        return json.loads(resp.text)["response"]["result"]
