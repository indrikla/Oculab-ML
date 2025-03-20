import threading
import requests
from ultralytics import YOLO
from PIL import Image, ImageDraw
from flask_restful import Resource, reqparse
from flask import request, send_file
from skimage.metrics import structural_similarity as ssim
import io
import cv2
import numpy as np
import tempfile
import os
import base64
from bson import Binary
from io import BytesIO
import aiohttp
import constants
import asyncio
import pymongo
import gridfs

from utils.uploadStorage import upload_pil_image_to_s3_public

model = YOLO(constants.DECENT_NOT_DECENT_MODEL)
objectDetectModel = YOLO(constants.OBJECT_DETECT_MODEL)

# Ensure the output directories exist
os.makedirs('images/decent', exist_ok=True)
os.makedirs('images/not decent', exist_ok=True)
os.makedirs('images/frames_output', exist_ok=True)

parser = reqparse.RequestParser()
parser.add_argument('examinationId', type=str, required=True, help="examinationid cannot be blank")
# Class that handles from video to eliminate blurry frames with YOLO
# and return array of frames that are not blurry and decent
class YOLOFramesElimination(Resource):

    # def __init__(self):
    #     # Connect to MongoDB
    #     self.client = pymongo.MongoClient("mongodb+srv://mrasyadc:EMI2Mbx9BPzsP033@cluster0.pivgh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    #     self.db = self.client["images"]  # Replace with your database name
    #     self.fs = gridfs.GridFS(self.db)

    def post(self, examinationId):
        try:
            # Check if a video is uploaded
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

            # Run the processing in a background thread and immediately return response
            thread = threading.Thread(target=self.run_async_in_thread, args=( examinationId, temp_video_path))
            thread.start()

            # Immediately return response while processing runs in background
            return "loading...", 200

        except Exception as e:
            # Log any errors and return an error message
            print(f"Error processing the video: {e}")
            return {'error': str(e)}, 500
        
    def run_async_in_thread(self, examinationId, video_path):
        """Runs video processing asynchronously in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Make sure the function is awaited properly
        loop.run_until_complete(self.process_video(examinationId, video_path))
        loop.close()

    async def process_video(self, examinationId, video_path):
        """Process the video file asynchronously."""
        try:
            # Open the video file
            video = cv2.VideoCapture(video_path)
            frames = []
            total_frames = 0

            # Read the video frame by frame
            while video.isOpened():
                success, image = video.read()
                if not success:
                    break

                # Process the image with YOLO, if successful, add to the frames array
                result = model(image)
                getProbs = result[0].probs.data.tolist()  # Make sure 'probs' is a valid field
                max_index = np.argmax(getProbs)
                classification = result[0].names[max_index]  # Make sure 'names' is valid
                total_frames += 1

                # Check if result classification is decent
                if classification == 'decent':
                    frames.append(image)

            # Close the video file
            video.release()
            print(f"total frames: {total_frames}")

            # If no decent frames, return error
            if len(frames) == 0:
                return {'error': 'No decent frames found'}, 400

            # Filter out similar frames
            final_frames = self.filter_similar_frames(frames)

            # Perform object detection and IAUTLD scoring
            results_dict = await self.process_frames_for_detection(examinationId, final_frames)

            # Run save_results_async to save results to the backend
            await save_results_async(examinationId, results_dict)

            # Save processed frames to video
            output_path = "images/frames_output/output_video.mp4"
            self.save_frames_to_video(final_frames, output_path)

        except Exception as e:
            print(f"Error processing video in async thread: {e}")


    def detect_bacteria(self, frame, frame_number):
        """Performs YOLO object detection on a single frame to count bacteria and extract confidence levels."""
        # Convert OpenCV frame to PIL image for YOLO detection
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform YOLO object detection for bacteria
        detection_results = objectDetectModel.predict(pil_image)

        # Extract bacteria count and confidence levels
        bacteria_count = len(detection_results[0].obb)
        confidence_levels = [box.conf.item() for box in detection_results[0].obb]
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0

        # Convert the PIL image to bytes for MongoDB storage as JPEG with reduced quality
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=constants.IMAGE_QUALITY)
        img_byte_arr = img_byte_arr.getvalue()

        # Save original image to MongoDB
       

        # Draw bounding boxes on the frame
        draw = ImageDraw.Draw(pil_image)
        for box in detection_results[0].obb:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extract bounding box coordinates
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)  # Draw red bounding box

        # # Convert the PIL image with bounding boxes to bytes for MongoDB storage as JPEG with reduced quality
        # img_byte_arr_bb = BytesIO()
        # pil_image.save(img_byte_arr_bb, format='JPEG', quality=85)  # Save as JPEG with quality 85
        # img_byte_arr_bb = img_byte_arr_bb.getvalue()

        # TODO: Save bounding box image to MongoDB
        # TODO: Or to Image Storage and return its url
        # original_image_id = self.fs.put(img_byte_arr, filename=f"original_frame_{frame_number}.jpg")
        # bounding_box_image_id = self.fs.put(img_byte_arr_bb, filename=f"bounding_box_frame_{frame_number}.jpg")

        bounding_box_image_url = upload_pil_image_to_s3_public(pil_image, 'oculab-fov')

        # create new variable types for BTA0, BTA1TO9, BTAABOVE9 based on bacteria_count
        bacteria_count_types = ""
        if bacteria_count == 0:
            bacteria_count_types = "0 BTA"
        elif 1 <= bacteria_count <= 9:
            bacteria_count_types = "1-9 BTA"
        else:
            bacteria_count_types = "≥ 10 BTA"


        # Return the frame number, detection results, and image IDs
        return {
            "order": frame_number,
            "systemCount": bacteria_count,
            "confidenceLevel": avg_confidence,
            "individual_confidences": confidence_levels,
            "type": bacteria_count_types,

            # TODO: Uncomment these lines to store image IDs in the results
            # "original_image_id": str(original_image_id),  # Store ObjectId as string
            # "bounding_box_image_id": str(bounding_box_image_id)  # Store ObjectId as string

            # TODO: Uncomment these lines to store image urls in the results
            # "original_image_url": f"url_to_original_image",
            "bounding_box_image_url": bounding_box_image_url
        }
        

    async def process_frames_for_detection(self, examinationId, frames):
        """Processes the filtered frames for bacteria detection and calculates IAUTLD score."""
        results_dict = {"IAUTLD_score": "", "confidence_level_aggregated": 0, "images": {}}
        images_bacteria_counts = []
        all_confidence_levels = []

        for i, frame in enumerate(frames):
            if i == 100: # Limit to 100 frames sesuai IUATLD
                break
            # Detect bacteria in each frame
            detection_result = self.detect_bacteria(frame, i + 1)

            # Store detection results
            # results_dict["images"][f'frame_{i + 1}'] = {
            #     "bacteria_count": detection_result["bacteria_count"],
            #     "confidence_levels": detection_result["confidence_levels"],
            #     # "original_image_id": detection_result["original_image_id"],
            #     # "bounding_box_image_id": detection_result["bounding_box_image_id"]
            # }

            # send fov to backend
            
            fov_dict = {
                "order": detection_result["order"],
                "systemCount": detection_result["systemCount"],
                "confidenceLevel": detection_result["confidenceLevel"],
                "type": detection_result["type"],

                # TODO: Uncomment these lines to store image IDs in the results
                # "original_image_id": detection_result["original_image_id"],
                # "bounding_box_image_id": detection_result["bounding_box_image_id"]

                # TODO: Uncomment these lines to store image urls in the results
                # "original_image_url": detection_result["original_image_url"],
                "image": detection_result["bounding_box_image_url"]
            }
            await save_fov_async(examinationId, fov_dict)

            all_confidence_levels.extend([conf for conf in detection_result["individual_confidences"] if conf > 0])
            images_bacteria_counts.append(detection_result["systemCount"])

        # IAUTLD score determination
        total_bacteria_count = sum(images_bacteria_counts)
        count_1_to_9 = sum(1 for count in images_bacteria_counts if 1 <= count <= 9)
        count_greater_than_10 = sum(1 for count in images_bacteria_counts if count >= 10)

        if total_bacteria_count == 0:
            results_dict["IAUTLD_score"] = "NEGATIVE"
        elif count_1_to_9 >= 50:
            results_dict["IAUTLD_score"] = "Plus2"
        elif count_greater_than_10 >= 20:
            results_dict["IAUTLD_score"] = "Plus3"
        elif total_bacteria_count <= 9:
            results_dict["IAUTLD_score"] = "SCANTY"
        elif total_bacteria_count <= 99:
            results_dict["IAUTLD_score"] = "Plus1"
        else:
            results_dict["IAUTLD_score"] = "Plus1"

        # Aggregated confidence level across all frames
        results_dict["confidence_level_aggregated"] = sum(all_confidence_levels) / len(all_confidence_levels) if all_confidence_levels else 0

        # remove the images from the results_dict
        del results_dict["images"]

        # adjust name for IAUTLD_score to systemGrading, confidence_level_aggregated to confidenceLevelAggregated, total_bacteria_count to systemBacteriaTotalCount
        results_dict["systemGrading"] = results_dict.pop("IAUTLD_score")
        results_dict["confidenceLevelAggregated"] = results_dict.pop("confidence_level_aggregated")
        results_dict["systemBacteriaTotalCount"] = total_bacteria_count

        print(results_dict)

        return results_dict
    
    def save_final_frames_to_folder_not_similar(self, frames):
        """Saves final frames to a folder."""
        if len(frames) == 0:
            return
        
        print(f"total final frames: {len(frames)}")
        
        for i, frame in enumerate(frames):
            cv2.imwrite(f'images/not similar/{i}.png', frame)

    def filter_similar_frames(self, frames):
        """Filters out frames that are too similar to each other using SSIM."""
        filtered_frames = [frames[0]]  # Always keep the first frame
        
        for i in range(1, len(frames)):
            last_frame = filtered_frames[-1]
            current_frame = frames[i]
            
            # Convert frames to grayscale for SSIM comparison
            gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM between the two frames
            score, _ = ssim(gray_last, gray_current, full=True)

            print(f"SSIM score: {score}")
            
            # If frames are not similar, keep the current frame
            if score < constants.SSIM_THRESHOLD:  # NOTE: Adjust threshold as needed
                filtered_frames.append(current_frame)
        
        return filtered_frames
    
    def save_frames_to_video(self, frames, output_path='images/output.mp4'):
        """Saves frames to a video file."""
        if len(frames) == 0:
            return
        
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()


async def save_results_async(examinationId, results_dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{constants.BACKEND_URL_EXPRESS["save_results"]}/{examinationId}', json=results_dict) as response:
            response_json = await response.json(content_type=None)
            print("saving data is finished with: response", response_json.get('status'))
            print("saving data is finished with: response", response_json.get('data'))
            # return

async def save_fov_async(examinationId, fov_dict):
    async with aiohttp.ClientSession() as session:
        print(f'{constants.BACKEND_URL_EXPRESS["save_fov"]}/{examinationId}')
        async with session.post(f'{constants.BACKEND_URL_EXPRESS["save_fov"]}/{examinationId}', json=fov_dict) as response:
            response_json = await response.json(content_type=None)  
            print("saving data is finished with: response", response_json.get('status'))
            print("saving data is finished with: response", response_json.get('data'))
            # return

def allowed_video_file(filename):
    # Add any allowed extensions you want to accept
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mov', 'mp4', 'avi'}

