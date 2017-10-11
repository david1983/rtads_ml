import os
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS']= "gcloud.json"

cli = storage.Client()
bucket = cli.get_bucket('ml-data-bucket')

filename = "5763210187636736/5636904326266880/SagePayReport1503308255095.csv"



# print(blob.download_as_string())
