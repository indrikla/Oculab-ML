
import tempfile
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask_restful import Resource, reqparse
from flask import request, send_file
import io
import pprint
import os
import constants

from utils.uploadStorage import upload_pil_image_to_s3_public

# Load YOLOv8 model (replace 'best.pt' with your custom model if needed)
model = YOLO(constants.OBJECT_DETECT_MODEL)

# Ensure the output directories exist
os.makedirs('images/decent', exist_ok=True)
os.makedirs('images/not decent', exist_ok=True)
os.makedirs('images/frames_output', exist_ok=True)

parser = reqparse.RequestParser()
parser.add_argument('examinationid', type=str, required=True, help="examinationid cannot be blank")
# Class that handles from video to eliminate blurry frames with YOLO
# and return array of frames that are not blurry and decent

class CheckVideo(Resource):
    def post(self, examinationId):
        try:
            if 'video' not in request.files:
                return {'error': 'No video provided'}, 400
            
            # Check if examinationId is exist in params

            if not examinationId:
                return {'error': 'No examinationId provided'}, 400
            
            # Read the video file
            video_file = request.files['video']

            # Ensure the uploaded file is a video
            if not video_file or not allowed_video_file(video_file.filename):
                return {'error': 'Invalid video file'}, 400
            
            # Create a temporary file to save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
                temp_video_file.write(video_file.read())
                temp_video_path = temp_video_file.name
            
            return {"video_path": temp_video_path}, 200

        except Exception as e:
            # Log any errors and return an error message
            print(f"Error processing the image: {e}")
            return {'error': str(e)}, 500

def allowed_video_file(filename):
    # Add any allowed extensions you want to accept
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mov', 'mp4', 'avi'}