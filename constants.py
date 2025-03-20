import os

BE_ENDPOINT_URL = os.getenv('BE_ENDPOINT_URL', 'http://localhost:3000')

BACKEND_URL_EXPRESS = {
    'save_results': f'{BE_ENDPOINT_URL}/systemResult/post-system-result',
    # /systemResult/post-system-result/:examinationId
    'save_fov': f'{BE_ENDPOINT_URL}/fov/post-fov-data'
    # /post-fov-data/:examinationId
}

SSIM_THRESHOLD = 0.9

IMAGE_QUALITY = 85

OBJECT_DETECT_MODEL = 'machine-learning/model-obb-12-nov-2024-yolo-8-s.pt'
DECENT_NOT_DECENT_MODEL = 'machine-learning/model-frame-decent.pt'