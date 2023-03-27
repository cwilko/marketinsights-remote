import requests
import json


def get(url, headers, debug=False):
    return sanitise(requests.get(url=url, headers=headers, timeout=300), url, debug)


def put(url, headers, data, debug=False):
    return sanitise(requests.put(url=url, headers=headers, data=data, timeout=300), url, debug)


def post(url, headers, data, debug=False):
    return sanitise(requests.post(url=url, headers=headers, data=data, timeout=300), url, debug)


def delete(url, headers, debug=False):
    return sanitise(requests.delete(url=url, headers=headers, timeout=300), url, debug)


def sanitise(raw, url, debug):

    if debug:
        print(url)
        print(raw)
        # print(raw.text[:100])
        # print(raw.json())

    response = json.loads(raw.text)

    if debug:
        print(raw.url)
        print("RESPONSE " + str(len(response)) + " length")

    if ("httpCode" in response) and (response["httpCode"] is not "200"):
        print("ERROR: INVALID HTTP CODE")
        raise Exception("Error in http response: ", url, response)

    return response
