import grpc
import os


class GRPCClient:

    def __init__(self, secret):
        self.connection = self.get_connection(secret["host"], secret["port"], secret["cert_path"])

    def get_connection(self, host, port, crt_path):
        """Establish secure grpc channel"""
        crt_path = os.path.expanduser(crt_path)
        with open(crt_path, 'rb') as f:
            trusted_certs = f.read()

        credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
        channel = grpc.secure_channel('{}:{}'.format(host, port), credentials)
        return channel

    def getChannel(self):
        return self.connection
