from google.cloud import datastore

dc = datastore.Client()


class Projects:
      def __init__(self):
            self.kind = "projects"

      def create(self, data):
            newKey = dc.key(self.kind)
            project = datastore.Entity(key=newKey)
            return dc.put(project)

      def read(self, id):
            entityKey = dc.Key(self.kind, id)
            project = datastore.Entity(key=entityKey);
            return project.get()

