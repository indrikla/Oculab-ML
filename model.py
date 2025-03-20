
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask_restful import Resource
from flask import request, send_file
import io
import pprint
import constants

from utils.uploadStorage import upload_pil_image_to_s3_public

# Load YOLOv8 model (replace 'best.pt' with your custom model if needed)
model = YOLO(constants.OBJECT_DETECT_MODEL)

class ObjectDetection(Resource):
    def post(self):
        try:
            # Check if an image is uploaded
            if 'image' not in request.files:
                return {'error': 'No image provided'}, 400
            
            # Read the image file
            image_file = request.files['image']

            # Ensure the uploaded file is an image
            if not image_file or not allowed_file(image_file.filename):
                return {'error': 'Invalid image file'}, 400
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_file.read()))
            
            # Run YOLOv8 inference
            results = model.predict(image)

            # pprint.pprint(results[0].obb)
            # return "a", 200

            # Check if results contain boxes
            if not results[0].obb:
                return {'message': 'No objects detected'}, 200
            
            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)
            for box in results[0].obb:  # Iterate over detected objects
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extract bounding box coordinates
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # Draw red bounding box

            # upload_pil_image_to_s3_public(image, 'oculab-fov')  # Upload the image to S3

            # Save the image with bounding boxes into an in-memory buffer
            img_io = io.BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)  # Important: reset the buffer position to the beginning

            # Return the image
            return send_file(img_io, mimetype='image/jpeg')

        except Exception as e:
            # Log any errors and return an error message
            print(f"Error processing the image: {e}")
            return {'error': str(e)}, 500

def allowed_file(filename):
    # Add any allowed extensions you want to accept
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}