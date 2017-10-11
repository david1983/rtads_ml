import services.apierrors    as apierrors
import numpy                 as np
from services.storage import read_file, write_file
from sklearn.neighbors       import LocalOutlierFactor
from flask                   import Blueprint, request
import json

lofBP = Blueprint("lofBP", __name__)

@lofBP.route("/lof")
def root():
    return json.dumps({
        "name": "lof",
        "type": "cluster",
        "description": "algorithm is optional or a choice between",
        "params": ["neighbours", "algorithm"]
    })

@lofBP.route("/knn/fit", methods=['POST'])
def fit():
    req=request.get_json()
    neighburs=2
    if("params" in req):
        neighburs = req["params"]["neighbours"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if(user_id == None or project_id == None or filename == None):  return apierrors.NoData()
    else:
        return apierrors.NoData();

    fullPath = user_id + "/"+project_id+"/" + filename
    dataset = read_file(fullPath)
    clf = LocalOutlierFactor(n_neighbors=neighburs)

    y_pred = clf.fit_predict(dataset)
    y_pred_outliers = y_pred

    # plot the level sets of the decision function
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    resultObj = {
        "original": dataset,
        "outliers": y_pred_outliers,
        "decision_function": Z
    }

