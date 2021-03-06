# Set google json creadentials as environment variable
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= "gcloud.json"

# Import libraries
import matplotlib
matplotlib.use('Agg')
import base64
import json
import mwares.auth          as authmw
import services.apierrors   as apierrors
import pandas               as pd
from flask              import Flask, request
from algorithms         import dbscan, svm, knn, lof,pca
from services.storage   import read_file, write_file, write_base64_img
from io                 import StringIO,BytesIO
from matplotlib         import pyplot as plt
from sklearn.preprocessing import LabelEncoder as le

# instantiate a new Flask application
app = Flask(__name__)

# Configure CORS middleware
@app.after_request 
def after_request(response):
    header = response.headers    
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    header['Access-Control-Allow-Headers'] = "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With, auth"
    return response

# Configure authentication method
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


def plot(plot):     
    image = BytesIO()    
    fig = plot.get_figure()                   
    fig.savefig(image, format='png')
    image.seek(0)  # rewind to beginning of file
    figdata_png = image.getvalue()     
    d = base64.encodestring(image.getvalue()).decode('ascii')   
    return '%s' % (d)

@app.route('/analyse', methods=['POST'])
def analyse():
    from pandas.plotting import scatter_matrix
    req=request.get_json()
    user_id = req["params"]["user_id"]
    project_id = req["params"]["project_id"]
    filename = req["params"]["filename"]       
    fullPath = user_id + "/"+project_id+"/" + filename
    dataset_file = read_file(fullPath)    
    if(dataset_file==None): return apierrors.ErrorMessage("dataset not found")
    
    
    file = StringIO(dataset_file.decode('utf-8'))
    dataset = pd.read_csv(file)     
    if "label_encode" in req:
        dataset = pd.read_csv(file, dtype="unicode")             
        dataset = dataset.apply(le().fit_transform)
    dataset = dataset.fillna(0)

    hp = plt.subplot()
    dataset.hist(ax=hp, figsize=(12,12))
    dp = dataset.plot(kind='density')
    bp = dataset.plot(kind='box')
    sm = scatter_matrix(dataset, figsize=(12,12)) 

    resultset = {
        "plot":    write_base64_img(user_id,project_id,"plot.png",plot(dataset.plot())),
        "hp_plot": write_base64_img(user_id,project_id,"hp.png",plot(hp)),
        "dp_plot": write_base64_img(user_id,project_id,"dp.png",plot(dp)),
        "bp_plot": write_base64_img(user_id,project_id,"bp.png",plot(bp)),
        "sm_plot": write_base64_img(user_id,project_id,"sm.png",plot(sm[0][0]))
    }
    return json.dumps(resultset)

# handle errors
@app.errorhandler(Exception)
def all_exception_handler(error):
    print(error)
    return json.dumps({"error": str(error)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return json.dumps({"error": "route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    print(e)
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='0.0.0.0', port=8080, debug=True)
