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
from services.storage       import read_file, write_file, get_pickle
from models.projects        import Projects

dbscanBP = Blueprint("dbscanBP", __name__)


@dbscanBP.route("/dbscan")
def root():
    return json.dumps({"name": "dbscan", "type": "clustering"})



def preProcess(dataset):
    from sklearn.preprocessing  import StandardScaler
    from sklearn                import preprocessing
    le = preprocessing.LabelEncoder()
    X = dataset.apply(le.fit_transform)
    X = StandardScaler().fit_transform(X)            
    return X


@dbscanBP.route("/dbscan/fit", methods=['POST'])
def fit():
    req = request.get_json()
    eps=0.7
    min_samples=4
    if("params" in req):
        eps = float(req["params"]["eps"])
        min_samples = float(req["params"]["min"])
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
    
    X = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = preProcess(dataset=X)
    print(type(X[0][0]))
    DB = DBSCAN(eps, min_samples)   
    s = pickle.dumps(DB)
    write_file(user_id, project_id, "pickle.pkl", s) 
    db = DB.fit(X)    
    print("ok")
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
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

    P = Projects(user_id, project_id)
    project = P.read(id=project_id)

    P.addDataset(data)
    # fullPath = user_id + "/"+project_id+"/" + project.fileName
    # dataset = read_file(fullPath)
    # if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    # le = preprocessing.LabelEncoder()
    # X = pd.read_csv(StringIO(dataset.decode('utf-8'))).tail(200)

    dataset = P.getDataset()
    X = []
    keys = []
    
    for i in dataset:       
        a = json.dumps(i["data"])
        b = json.loads(a)
        X.append(b)
        
        # X.append()

    X = pd.DataFrame(X)
    print(X.head())
    X = preProcess(dataset=X)
    pkl_file = get_pickle(user_id + "/"+project_id+"/pickle.pkl")    
    model = pickle.load(pkl_file)
    db = model.fit(X)    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    resultObj = {
        "clusters": n_clusters_,
        "dataset": X.tolist(),
        "labels": labels.tolist()
    }
    # return json.dumps(resultObj)

    return json.dumps(resultObj)