from google.cloud import storage
from datetime import timedelta
try:
   import urllib2
except ImportError:
   import urllib.request as urllib2

client = storage.Client()
bucket = client.get_bucket('ml-data-bucket')

def read_file(file_path):
   blob = bucket.get_blob(file_path)
   print(blob)
   if(blob == None): return None
   day = timedelta(hours=1)
   signedUrl = blob.generate_signed_url(day)
   response = urllib2.urlopen(signedUrl)
   return response.read()

def write_file(user_id, project_id, file_name, file_content):
   blob = bucket.blob(user_id + '/'+project_id+'/' + file_name)
   blob.upload_from_string(file_content)
   return