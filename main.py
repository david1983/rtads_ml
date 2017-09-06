import logging
import json
from flask import Flask, request

import algorithms.dbscan as dbscan
import algorithms.svm as svm

app = Flask(__name__)

app.register_blueprint(dbscan.dbscanBP)
app.register_blueprint(svm.svmBp)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return json.dumps({"version": 1})

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
