from google.cloud import datastore
from datetime import datetime
import json

dc = datastore.Client("rtads-179207")


class Projects:
      def __init__(self, user_id, project_id):
            self.kind = "projects"
            self.ds_kind = "datasets"
            self.user_id = user_id
            self.project_id = project_id

      def create(self, data):
            newKey = dc.key(self.kind)
            project = datastore.Entity(key=newKey)
            return dc.put(project)

      def read(self, id):
            entityKey = dc.key(self.kind, id)
            entity = datastore.Entity(key=entityKey)
            print(entity)
            return entity.get(entityKey)

      def getDataset(self, limit=100):
            query = dc.query(kind=self.ds_kind)
            query.add_filter('projectId', '=', self.project_id)        
            query.add_filter('userId', '=', self.user_id)   
            query.order = "-createdAt"
            print(query)
            return list(query.fetch(limit=limit))

      def addDataset(self, data):
            newKey = dc.key(self.ds_kind)
            dataset = datastore.Entity(key=newKey)
            dataset.update({
                  'data': json.dumps(data),
                  'projectId': self.project_id,
                  'userId': self.user_id,
                  'createdAt': str(datetime.now()),
                  'updatedAt': str(datetime.now())
            })
            # dataset["data"] = data
            # dataset["projectId"] = self.project_id
            # dataset["userId"] = self.user_id
            # dataset["createdAt"] = str(datetime.now())
            # dataset["updatedAt"] = str(datetime.now())

            return dc.put(dataset)