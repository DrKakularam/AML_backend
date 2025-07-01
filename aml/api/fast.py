from fastapi import FastAPI, File, UploadFile
from aml.preprocess.model import load_model_local
from aml.preprocess.preprocess import preprocess_image
import io
from PIL import Image
import numpy as np

app = FastAPI()
model=load_model_local()

@app.post("/classify/")
async def classify_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))

    # return str(type(pil_image))
    img_array = preprocess_image(pil_image)
    # return img_array.shape
    prediction = model.predict(img_array)
    # predicted_class = np.argmax(prediction, axis=1)[0]
    return str(prediction)
