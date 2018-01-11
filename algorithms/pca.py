import json
import services.apierrors   as apierrors
from services.storage       import read_file, write_file
from sklearn.decomposition  import PCA
from flask                  import Blueprint, request

pcaBP = Blueprint("pcaBP", __name__)

@pcaBP.route("/pca")
def root():
    return json.dumps({"name": "pca", "type": "orthogonal linear transformation"})

@pcaBP.route("/pca/transform", methods=['POST'])
def fit():
    req=request.get_json()
    dimensions=2
    if("params" in req):
        dimensions = req["params"]["dimensions"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if(user_id == None or project_id == None or filename == None):  return apierrors.NoData()
    else:
        return apierrors.NoData();

    fullPath = user_id + "/"+project_id+"/" + filename
    dataset = read_file(fullPath)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")

    pca = PCA(n_components=dimensions)
    transformed = pca.fit_transform(dataset)
    resultObj = {
        "original": dataset,
        "transformed": transformed
    }
    # resultObj = {}
    return json.dumps(resultObj)
