import pickle
import json
import pandas               as pd
import numpy                as np
import services.apierrors   as apierrors
from io                     import StringIO
from sklearn.cluster        import DBSCAN
from sklearn.preprocessing  import StandardScaler
from sklearn                import preprocessing
from flask                  import Blueprint, request
from services.storage       import read_file, write_file
from models.projects        import Projects

dbscanBP = Blueprint("dbscanBP", __name__)


@dbscanBP.route("/dbscan")
def root():
    return json.dumps({"name": "dbscan", "type": "clustering"})


@dbscanBP.route("/dbscan/fit", methods=['POST'])
def fit():
    req = request.get_json()
    eps=0.7
    min_samples=4
    if("params" in req):
        eps = req["params"]["eps"]
        min_samples = req["params"]["min"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if(user_id == None or project_id == None or filename == None):  return apierrors.NoData()
        if "max" in req:
            max_samples = req["params"]["max"]
    else:
        return apierrors.NoData();


    fullPath = user_id + "/"+project_id+"/" + filename
    dataset = read_file(fullPath)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    le = preprocessing.LabelEncoder()
    X = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = X.fillna(0)
    X = X.apply(le.fit_transform)
    X = StandardScaler().fit_transform(X)

    DB = DBSCAN(eps, min_samples)
    s = pickle.dumps(DB)    
    write_file(user_id, project_id, "pickle.pkl", s)
    db = DB.fit(X)    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    resultObj = {
        "clusters": n_clusters_,
        "dataset": X.tolist(),
        "labels": labels.tolist()
    }
    # resultObj = {}
    return json.dumps(resultObj)


@dbscanBP.route("/dbscan/predict", methods=['POST'])
def predict():
    req = request.get_json()
    if("params" in req):            
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]        
        data = req["params"]["data"]
    else:
        return apierrors.ErrorMessage("You need to specify parameters to load ")

    project = Projects.read(id=project_id)
    fullPath = user_id + "/"+project_id+"/" + project.fileName
    dataset = read_file(fullPath)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    le = preprocessing.LabelEncoder()
    X = pd.read_csv(StringIO(dataset.decode('utf-8'))).tail(200)
    X = X.fillna(0)
    X = X.apply(le.fit_transform)    
    X = StandardScaler().fit_transform(X)

    newData = pd.DataFrame(data);
    X.append(newData)

    pkl_file = read_file(user_id + "/"+project_id+"/pickle.pkl")
    if(pkl_file==None): return apierrors.ErrorMessage("No pickle file found, maybe you should train the model first")
    model = pickle.load(StringIO(pkl_file.decode('utf-8')))
    model.fit(X)    
    resultObj = {
        "clusters": n_clusters_,
        "dataset": X.tolist(),
        "labels": labels.tolist()
    }
    return json.dumps(resultObj)