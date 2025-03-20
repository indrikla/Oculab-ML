from flask import Flask, request
from flask_restful import Api, Resource
from ultralytics import YOLO
from PIL import Image
import io

import constants

model = YOLO(constants.OBJECT_DETECT_MODEL)


class BacteriaDetection(Resource):
    def post(self):
        try:
            if 'images' not in request.files:
                return {'error': 'No images provided'}, 400

            # Retrieve multiple image files from request
            files = request.files.getlist('images')
            results_dict = {"IAUTLD_score": "", "confidence_level_aggregated": 0, "images": {}}
            images_bacteria_counts = []
            all_confidence_levels = []  # Store all confidence levels for aggregation

            # Iterate through each image file
            for file in files:
                if not allowed_file(file.filename):
                    return {'error': f'Invalid file format: {file.filename}'}, 400

                # Open the image file
                image = Image.open(io.BytesIO(file.read())).convert("RGB")

                # Perform YOLO object detection on the image
                results = model.predict(image)

                # Extract bacteria count and confidence levels
                bacteria_count = len(results[0].obb)  # Count detected boxes (assumed to represent bacteria)

                # Extract confidence levels
                confidence_levels = [box.conf.item() for box in results[0].obb]  # Extract confidence levels as floats

                if confidence_levels:
                    avg_confidence = sum(confidence_levels) / len(confidence_levels)
                else:
                    avg_confidence = 0  # No detections, set average confidence to 0

                # Store results for each image
                results_dict["images"][file.filename] = {
                    "bacteria_count": bacteria_count,
                    "confidence_levels": avg_confidence  # Include all confidence levels
                }

                # Aggregate confidence levels for non-zero detections
                all_confidence_levels.extend([conf for conf in confidence_levels if conf > 0])
                images_bacteria_counts.append(bacteria_count)

            # Calculate total bacteria count
            total_bacteria_count = sum(images_bacteria_counts)

            # Initialize counters for IAUTLD score determination
            count_1_to_9 = sum(1 for count in images_bacteria_counts if 1 <= count <= 9)
            count_greater_than_10 = sum(1 for count in images_bacteria_counts if count >= 10)

            # Determine IAUTLD score based on the new logic
            if total_bacteria_count == 0:
                results_dict["IAUTLD_score"] = "Negative"
            elif count_1_to_9 >= 50:
                results_dict["IAUTLD_score"] = "2+ (Positif dua)"
            elif count_greater_than_10 >= 20:
                results_dict["IAUTLD_score"] = "3+ (Positif tiga)"
            elif total_bacteria_count <= 9:
                results_dict["IAUTLD_score"] = f"Scanty (Total: {total_bacteria_count} BTA found)"
            elif total_bacteria_count <= 99:
                results_dict["IAUTLD_score"] = "1+ (Positif satu)"
            else:
                results_dict["IAUTLD_score"] = "1+ (Positif satu)"  # Default case

            # Calculate aggregated confidence level across all images
            if all_confidence_levels:
                results_dict["confidence_level_aggregated"] = sum(all_confidence_levels) / len(all_confidence_levels)
            else:
                results_dict["confidence_level_aggregated"] = 0

            # Return the results as a dictionary
            return results_dict, 200

        except Exception as e:
            # Log any errors and return an error message
            print(f"Error processing the images: {e}")
            return {'error': str(e)}, 500


def allowed_file(filename):
    """Function to check allowed file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}
