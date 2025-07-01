from aml.params import *
from aml.preprocess.model import load_model_local
from PIL import Image
import tensorflow as tf
import numpy as np
from google.cloud import storage


def preprocess_image(image):
    "Returns prediction for an image"
    # model=load_model_local()
    # img = Image.open(image)
    img = image.convert('RGB')        # Ensure 3 channels
    img = img.resize((144, 144))    # Resize as needed
    arr = np.array(img).astype(np.float32) / 255.0
    arr=np.expand_dims(arr, axis=0)
    return arr

def preprocess_image_path(image_path):
    "Returns prediction for an image"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(image_path)
    if blob.exists():
        img_bytes = blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')        # Ensure 3 channels
        img = img.resize((144, 144))    # Resize as needed
        arr = np.array(img).astype(np.float32) / 255.0
        arr=np.expand_dims(arr, axis=0)
        return arr
    return None




from PIL import Image
import io

# gcs_image_path = "gs://mybucket/image.jpg"
bucket_name="aml_data_drkakularam"
image_path="aml_data/control/MPP_image_100.tif"
# Parse GCS path
# bucket_name = gcs_image_path.split("/")[2]
# blob_name = "/".join(gcs_image_path.split("/")[3:])
# aml_data_drkakularam/aml_data/control
# Download image from GCS
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(image_path)
if blob.exists():
    img_bytes = blob.download_as_bytes()
    img = Image.open(io.BytesIO(img_bytes))
else:
    print("Blob does not exist!")
# # Now you can use `img` as a PIL image object
img
