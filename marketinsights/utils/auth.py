import json
import os

# File based credentials store


class CredentialsStore():

    def __init__(self, context='~/.quantutils'):
        context = os.path.expanduser(context)
        if not os.path.exists(context):
            os.makedirs(context)
        self.context = context

    def getSecrets(self, key):
        with open("".join([self.context, "/", key, ".json"])) as infile:
            secrets = json.load(infile)
        return secrets

    def putSecrets(self, key, secrets):
        with open("".join([self.context, "/", key, ".json"]), 'w') as outfile:
            json.dump(secrets, outfile)
