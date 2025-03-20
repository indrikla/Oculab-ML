import os
import boto3
import io
import uuid
from PIL import Image
from botocore.exceptions import NoCredentialsError

def upload_pil_image_to_s3_public(image, bucket_name, format='PNG'):
    # Convert PIL Image to BytesIO
    image_io = io.BytesIO()
    image.save(image_io, format=format)  # Save image to BytesIO buffer
    image_io.seek(0)  # Reset the stream position to the beginning

    # Generate a unique object name using UUID
    object_name = f"{uuid.uuid4()}.{format.lower()}"  # Example: 'f47ac10b-58cc-4372-a567-0e02b2c3d479.png'


    AWS_ENDPOINT_URL = os.getenv('AWS_ENDPOINT_URL')
    AWS_BUCKET=os.getenv('AWS_BUCKET')




    # Initialize the S3 client
    s3 = boto3.client('s3')

    try:
        # Upload the file-like object
        s3.upload_fileobj(image_io, bucket_name, object_name, ExtraArgs={'ACL': 'public-read'})
        print(f"Image uploaded to {AWS_ENDPOINT_URL}/{AWS_BUCKET}/{object_name}")
        # return object_name  # Return the generated object name for reference
        # return the public URL
        return f"{AWS_ENDPOINT_URL}/{AWS_BUCKET}/{object_name}"

    except NoCredentialsError:
        print("Credentials not available")

# Example usage
# image = Image.open('path_to_your_image.jpg')  # Load your image
# bucket_name = 'your_s3_bucket_name'
# generated_object_name = upload_pil_image_to_s3(image, bucket_name)

# print(f"Generated S3 Object Name: {generated_object_name}")
