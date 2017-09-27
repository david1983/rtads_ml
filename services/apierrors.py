import json


def ErrorMessage(msg):
    return json.dumps({"error": True, "message": msg})


def NoData():
    return ErrorMessage("no data")

def NoAuthToken():
    return ErrorMessage("No auth token")