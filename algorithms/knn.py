import services.apierrors   as apierrors
from services.storage import read_file, write_file, get_pickle
from sklearn.neighbors import NearestNeighbors
from flask import Blueprint, request
from io                     import StringIO
import pandas as pd
import json
import pickle
import numpy as np
from models.projects        import Projects

knnBP = Blueprint("knnBP", __name__)


@knnBP.route("/knn")
def root():
    return json.dumps({
        "name": "knn",
        "type": "cluster",
        "description": "This is a supervised ML algorithm so you need to provide a labeled dataset",
        "params": ["neighbours", "algorithm"]
    })

def preProcess(dataset):
    from sklearn.preprocessing  import StandardScaler
    from sklearn                import preprocessing
    le = preprocessing.LabelEncoder()
    X = dataset.apply(le.fit_transform)
    X = StandardScaler().fit_transform(X)            
    return X

@knnBP.route("/nn/fit", methods=['POST'])
def fit():
    req = request.get_json()
    neighburs = 2
    algorithm = "ball_tree"
    metric = "euclidean"
    if ("params" in req):
        if("neighbours" in req["params"]): neighburs = req["params"]["neighbours"]
        if("algorithm" in req["params"]): algorithm = req["params"]["algorithm"]
        if("metric" in req["params"]): metric = req["params"]["metric"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if (user_id == None or project_id == None or filename == None):  return apierrors.NoData()
    else:
        return apierrors.NoData();

    fullPath = user_id + "/" + project_id + "/" + filename
    dataset = read_file(fullPath)
    rawX = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = preProcess(dataset=rawX)

    print(X)
    NN = NearestNeighbors(n_neighbors=neighburs, algorithm=algorithm, metric=metric)
    s = pickle.dumps(NN)
    write_file(user_id, project_id, "pickle.pkl", s)
    nbrs = nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(pd.DataFrame(X).describe())
    print(pd.DataFrame(indices).describe())
    data = rawX.to_json()
    indexes = pd.DataFrame(indices).to_json()
    return '{ "data": ' + data + ', "indexes": '+indexes+'}'

@knnBP.route("/nn/predict", methods=["POST"])
def predict():
    
    req=request.get_json()
    if ("params" in req):  
        data = req["params"]["data"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
        if (user_id == None or project_id == None or filename == None):  return apierrors.NoData()
    else:
        return apierrors.NoData();

    P = Projects(user_id, project_id)
    project = P.read(id=project_id)
    P.addDataset(data)
    dataset = P.getDataset()
    # reshape the dataset  to a dataframe like object
    X = {}
    for k in dataset[0]["data"]:
        X[k] = []
    for i in dataset:               
        obj = json.loads(json.dumps(i["data"]))
        for k in obj:
            X[k].append(obj[k])
    # convert the array to pandas dataframe
    rawX = pd.DataFrame(X)    
    X = preProcess(dataset=rawX)
    pkl_file = get_pickle(user_id + "/"+project_id+"/pickle.pkl")    
    if(pkl_file==None): return apierrors.ErrorMessage("No pickle file found, maybe you should train the model first")    
    model = pickle.load(pkl_file)
    print(model.__dict__)
    nbrs = model.fit(X)
    distances, indices = nbrs.kneighbors(X)    
    
    
    obj = [X[0]]
    # for i in X[0]:
    #     print(i)
    #     obj.append([i])
    print(obj)
    data = rawX.to_json()
    indexes = pd.DataFrame(indices).to_json()    
    kneighbors = nbrs.kneighbors(obj)

    print(kneighbors)
    return '{ "data": ' + data + ', "indexes": '+indexes+', "kneighbours": '+kneighbors+'}'