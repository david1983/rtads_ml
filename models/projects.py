from google.cloud import datastore

dc = datastore.Client()

class Projects:
   def __init__(self):
      self.kind="projects"

   def create(self, data):
      newKey = dc.key(self.kind)
      project = datastore.Entity(key=newKey)
      dc.put(project)