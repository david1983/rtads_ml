from google.cloud import storage
from datetime import timedelta
import base64
try:
   import urllib2
except ImportError:
   import urllib.request as urllib2

client = storage.Client()
bucket = client.get_bucket('rtads-ml-data-bucket')

def getSignedUrl(file_path):
   blob = bucket.get_blob(file_path)   
   if(blob == None): return None
   day = timedelta(hours=1)
   return blob.generate_signed_url(day)

def get_pickle(file_path):
   signedUrl = getSignedUrl(file_path)
   return urllib2.urlopen(signedUrl)

def read_file(file_path):
   signedUrl = getSignedUrl(file_path)
   response = urllib2.urlopen(signedUrl)
   return response.read()

def write_file(user_id, project_id, file_name, file_content):
   blob = bucket.blob(user_id + '/'+project_id+'/' + file_name)
   blob.upload_from_string(file_content)   
   day = timedelta(days=365)
   return blob.generate_signed_url(day)

def write_base64_img(user_id, project_id, file_name, file_content):
   blob = bucket.blob(user_id + '/'+project_id+'/' + file_name)
   blob.upload_from_string(base64.b64decode(file_content))   
   day = timedelta(days=365)
   return blob.generate_signed_url(day)

