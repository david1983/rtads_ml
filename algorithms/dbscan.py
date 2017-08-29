import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import metrics
from flask import Blueprint, request
import services.apierrors as apierrors


dbscanBP = Blueprint("dbscanBP", __name__)

@dbscanBP.route("/dbscan")
def root():
   return json.dumps({"name": "dbscan", "type": "clustering"})

@dbscanBP.route("/dbscan/fit", methods=['POST'])
def fit():   
   req = request.get_json();  
   if not "dataset" in req:
      return apierrors.NoData()

   eps = req["params"]["eps"]
   min_samples = req["params"]["min"]
   if "max" in req:
      max_samples = req["params"]["max"]
   X = req["dataset"]
   X = StandardScaler().fit_transform(X)
   db = DBSCAN(eps, min_samples).fit(X)
   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   core_samples_mask[db.core_sample_indices_] = True
   labels = db.labels_
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

   resultObj = {
      "clusters":  n_clusters_,      
      "dataset" : req["dataset"],
      "labels" : labels.tolist()
   }

   return json.dumps(resultObj)
