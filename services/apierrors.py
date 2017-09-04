import json


def ErrorMessage(msg):
    return json.dumps({"error": True, "message": msg})


def NoData():
    return ErrorMessage("no data")
