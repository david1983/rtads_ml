# Set google json creadentials as environment variable
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= "gcloud.json"

import logging
import json
from flask import Flask, request
from algorithms import dbscan, svm, knn, lof,pca
import mwares.auth as authmw
import services.apierrors as apierrors
import pandas as pd
from services.storage import read_file, write_file
from io                     import StringIO
from flask_cors import CORS, cross_origin
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import base64
# instantiate a new Flask application
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.after_request # blueprint can also be app~~
def after_request(response):
    header = response.headers    
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    header['Access-Control-Allow-Headers'] = "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With, auth"

    return response

# @app.before_request
# def before_request():
    # if request.headers["auth"]!="321":
    #     return apierrors.NoAuthToken()
    

# blueprints registration
app.register_blueprint(dbscan.dbscanBP)
app.register_blueprint(svm.svmBp)
app.register_blueprint(knn.knnBP)
app.register_blueprint(lof.lofBP)
app.register_blueprint(pca.pcaBP)

# global routes
@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return json.dumps({"version": 1})

from io import BytesIO
def plot(data):
    image = StringIO()    
    image = BytesIO()
    plot = data.plot()
    fig = plot.get_figure()          
    
    print(fig)  
    fig.savefig(image, format='png')
    image.seek(0)  # rewind to beginning of file
    figdata_png = image.getvalue()     
    d = base64.encodestring(image.getvalue()).decode('ascii')   
    return '%s' % (d)

@app.route('/analyse', methods=['POST'])
def analyse():
    req=request.get_json()
    user_id = req["params"]["user_id"]
    project_id = req["params"]["project_id"]
    filename = req["params"]["filename"]
    fullPath = user_id + "/"+project_id+"/" + filename
    dataset = read_file(fullPath)
    if(dataset==None): return apierrors.ErrorMessage("dataset not found")
    file = StringIO(dataset.decode('utf-8'))
    dataset = pd.read_csv(file)
    img = plot(dataset)
    resultset = {
        "plot": img
    }
    return json.dumps(resultset)
    

# handle errors
@app.errorhandler(Exception)
def all_exception_handler(error):
    return json.dumps({"error": str(error)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return json.dumps({"error": "route not found"}), 404



@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
