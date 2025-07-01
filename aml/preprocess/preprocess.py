from aml.params import *
from aml.preprocess.model import load_model_local
from PIL import Image
import tensorflow as tf
import numpy as np


def preprocess_image(image):
    "Returns prediction for an image"
    # model=load_model_local()
    # img = Image.open(image)
    img = image.convert('RGB')        # Ensure 3 channels
    img = img.resize((144, 144))    # Resize as needed
    arr = np.array(img).astype(np.float32) / 255.0
    arr=np.expand_dims(arr, axis=0)
    return arr
