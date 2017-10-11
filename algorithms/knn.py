import services.apierrors   as apierrors
from services.storage import read_file, write_file
from sklearn.neighbors import NearestNeighbors
from flask import Blueprint, request

knnBP = Blueprint("knnBP", __name__)


@knnBP.route("/knn")
def root():
    return json.dumps({
        "name": "knn",
        "type": "cluster",
        "description": " algorithm is optional or a choice between {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}",
        "params": ["neighbours", "algorithm"]
    })


@knnBP.route("/knn/fit", methods=['POST'])
def fit():
    neighburs = 2
    algorithm = "ball_tree"
    metric = "euclidean"
    if ("params" in req):
        neighburs = req["params"]["neighbours"]
        algorithm = req["params"]["algorithm"]
        metric = req["params"]["metric"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if (user_id == None or project_id == None or filename == None):  return apierrors.NoData()
    else:
        return apierrors.NoData();

    fullPath = user_id + "/" + project_id + "/" + filename
    dataset = read_file(fullPath)
    nbrs = NearestNeighbors(n_neighbors=neighburs, algorithm=algorithm, metric=metric).fit(dataset)
    distances, indices = nbrs.kneighbors(X)
    resultObj = {
        "original": dataset,
        "distances": distances,
        "indices": indices
    }
