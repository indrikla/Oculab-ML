import os
from flask import Flask, request, send_file
from flask_restful import Api, Resource
from checker import CheckConnection
import constants
from model import ObjectDetection
from modelAcceptLotsofImages import BacteriaDetection
from dummy import IPDetection
from YOLOFramesElimination import YOLOFramesElimination
from example import TodoList, Todo
from video import CheckVideo

app = Flask(__name__)
api = Api(app)

# Define API resource route
api.add_resource(CheckConnection, '/check-connection')
api.add_resource(CheckVideo, '/check-video/<examinationId>')
api.add_resource(IPDetection, '/ips')
api.add_resource(BacteriaDetection, '/detects')
api.add_resource(ObjectDetection, '/detect')
# curl -X POST -F "image=@images/bacteria2.jpg" http://127.0.0.1:5000/detect --output images/output.jpg
# tanda @ ini penting karena berfungsi untuk mengirimkan file ke server

api.add_resource(YOLOFramesElimination, '/extract-video/<examinationId>')

# curl -X POST -F "image=@images/bacteria2.jpg" http://127.0.0.1:5000/detect --output images/output.jpg
# tanda @ ini penting karena berfungsi untuk mengirimkan file ke server

# Example API
api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<todo_id>')


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)