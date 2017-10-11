import warnings
import json
import itertools
import pandas as pd
import numpy as np
import datetime as dt
from flask import Blueprint, request
import services.apierrors as apierrors
import statsmodels.api as sm
from sklearn import preprocessing
from io import StringIO
from services.storage import read_file, write_file

sarimaxBp = Blueprint("sarimaxBp", __name__)


@sarimaxBp.route("/svm")
def root():
    return json.dumps({
        "name":
            "Stochastic Autoregression Integrated Moving Average model with eXogenous inputs model ",
        "type":
            "regression"
    })


@sarimaxBp.route("/svm/fit", methods=['POST'])
def fit():
    req = request.get_json()
    if ("params" in req):
        eps = req["params"]["eps"]
        min_samples = req["params"]["min"]
        user_id = req["params"]["user_id"]
        project_id = req["params"]["project_id"]
        filename = req["params"]["project_id"]
        if (user_id == None or project_id == None or filename == None):
            return apierrors.NoData()
        if "max" in req:
            max_samples = req["params"]["max"]
    else:
        return apierrors.NoData()

    fullPath = user_id + "/" + project_id + "/" + filename

    dataset = read_file(fullPath)
    if (dataset == None): return apierrors.ErrorMessage("dataset not found")
    le = preprocessing.LabelEncoder()
    y = pd.read_csv(StringIO(dataset.decode('utf-8')))
    y = y.fillna(y.bfill())

    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12)
                    for x in list(itertools.product(p, d, q))]
    #test all different combinations of p d q parameters

    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    pdqparams = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    y,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

                results = mod.fit()
                pdqparams.append([param, param_seasonal, results.aic])
            except:
                continue

    for i in pdqparams:
        if(minval == None): minval = i[2]
        if (i[2] < minval):
            minval = i[2]
            minparams = i

    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=minparams[0],
                                    seasonal_order= minparams[1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()



    return ""
