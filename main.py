# Set google json creadentials as environment variable
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= "gcloud.json"

import logging
import json
from flask import Flask, request
import algorithms.dbscan as dbscan
import algorithms.svm as svm

# instantiate a new Flask application
app = Flask(__name__)

# blueprints registration
app.register_blueprint(dbscan.dbscanBP)
app.register_blueprint(svm.svmBp)


# global routes
@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return json.dumps({"version": 1})

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
