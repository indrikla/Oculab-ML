# this file is for try-and-error purposes of flask
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask_restful import Resource
from flask import request, send_file
import io

class CheckConnection(Resource):
    def get(self):
        """Endpoint to detect and return the IP address of the request."""
        client_ip = request.remote_addr  # Get the IP address of the client
        print(f"Client IP: {client_ip}")  # Print to console for debugging
        return {"connected": True,
                "client_ip": client_ip}, 200  # Return IP as a JSON response