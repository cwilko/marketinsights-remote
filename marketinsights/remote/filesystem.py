from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient

from marketinsights.utils.auth import CredentialsStore


class RemoteFS:

    def __init__(self, secret):
        self.creds = secret

    def put(self, sourcePath, targetPath=None, debug=False):

        if not targetPath:
            targetPath = self.creds["default_targetPath"]

        if debug:
            print(f"Tansferring {sourcePath} to {targetPath} on remote server, {self.creds['host']}")

        with SSHClient() as ssh:
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(hostname=self.creds["host"], username=self.creds["username"], password=self.creds["password"], allow_agent=False, look_for_keys=False)

            with SCPClient(ssh.get_transport()) as scp:
                scp.put(sourcePath, targetPath, recursive=True)

    def get(self, sourceFile, targetPath, sourcePath=None, debug=False):

        if not sourcePath:
            sourcePath = self.creds["default_targetPath"]

        if debug:
            print(f"Tansferring {sourcePath} to {targetPath} from remote server, {self.creds['host']}")

        with SSHClient() as ssh:
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(hostname=self.creds["host"], username=self.creds["username"], password=self.creds["password"], allow_agent=False, look_for_keys=False)

            with SCPClient(ssh.get_transport()) as scp:
                scp.get(f"{sourcePath}{sourceFile}", targetPath, recursive=True)
