import json
import pandas as pd
import numpy as np
import datetime as dt
from flask import Blueprint, request
import services.apierrors as apierrors
from sklearn import svm
from io import StringIO
from services.storage import read_file, write_file, get_pickle
import pickle
from models.projects        import Projects

svmBp = Blueprint("svmBp", __name__)


def preProcess(dataset):
    from sklearn.preprocessing  import StandardScaler
    from sklearn                import preprocessing
    le = preprocessing.LabelEncoder()
    X = dataset.apply(le.fit_transform)
    X = StandardScaler().fit_transform(X)            
    return X


@svmBp.route("/svm")
def root():
    return json.dumps({
        "name": "One-class Support Vector Machine",
        "type": "clustering"
    })


@svmBp.route("/svm/fit", methods=['POST'])
def fit():
    nu=0.1
    kernel="rbf"
    gamma=0.1    
    req = request.get_json()
    if "params" in req:
        if("nu" in req["params"]): nu = req["params"]["nu"]
        if("kernel" in req["params"]): kernel = req["params"]["kernel"]
        if("gamma" in req["params"]): gamma = req["params"]["gamma"]        
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
    else:
        return apierrors.NoData()

    print("start")
    fullPath = user_id + "/"+project_id+"/" + filename
    print(fullPath)
    dataset = read_file(fullPath)
    print(dataset)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    rawX = pd.read_csv(StringIO(dataset.decode('utf-8')))
    X = preProcess(dataset=rawX)
    X_train = X[0:int(len(X) * 0.66)]
    print("start")
    # fit the model
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    s = pickle.dumps(clf)
    write_file(user_id, project_id, "pickle.pkl", s)

    return json.dumps({          
        "dataset": json.loads(rawX.to_json()),
        "train_labels": y_pred_train.tolist(),        
        "labels": y_pred_test.tolist()
    })

@svmBp.route("/svm/predict", methods=['POST'])
def predict():
    req = request.get_json()
    if "params" in req:        
        data = req["params"]["data"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["filename"]
    else:
        return apierrors.NoData()
    print("start")
    P = Projects(user_id, project_id)
    project = P.read(id=project_id)
    P.addDataset(data)
    dataset = P.getDataset()    
    # reshape the dataset  to a dataframe like object
    X = {}
    if(type(dataset[0]["data"]) == str): dataset[0]["data"] = json.loads(dataset[0]["data"])
    
    for k in dataset[0]["data"]:        
        X[k] = []    

    for i in dataset:               
        if(type(i["data"]) == str): obj=json.loads(i["data"])
        else: obj=json.loads(json.dumps(i["data"]))
        for k in obj:
            X[k].append(obj[k])
    # convert the array to pandas dataframe
    
    rawX = pd.DataFrame(X)    
    
    X = preProcess(dataset=rawX)
    
    pkl_file = get_pickle(user_id + "/"+project_id+"/pickle.pkl")    
    if(pkl_file==None): return apierrors.ErrorMessage("No pickle file found, maybe you should train the model first")    
    
    model = pickle.load(pkl_file)        
    obj = [X[0]]     
    labels = pd.DataFrame(model.predict(X)).to_json()
    return json.dumps({
        "data": json.loads(rawX.to_json()),
        "labels": json.loads(labels)
    })