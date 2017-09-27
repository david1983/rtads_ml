import os
from flask import request
class Auth(object):

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):        
        return self.app(environ, start_response)