from fastapi import FastAPI, File, UploadFile
from aml.preprocess.model import load_model_local,load_model_gcs
from aml.preprocess.preprocess import preprocess_image, preprocess_image_path
import io
from PIL import Image
import numpy as np

app = FastAPI()
# app.state.model_local =load_model_local()
# app.state.model_cloud =load_model_gcs()
app.state.model =load_model_gcs()

@app.post("/upload_image/")
async def classify_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    img_array = preprocess_image(pil_image)
    model = app.state.model
    prediction = model.predict(img_array)
    # predicted_class = np.argmax(prediction, axis=1)[0]
    return {"prediction":prediction.tolist()}

@app.get("/image_dataset/")
async def classify_image(image_path: str):
    img_array = preprocess_image_path(image_path)
    if img_array is not  None:
        model = app.state.model
        prediction = model.predict(img_array)
        # predicted_class = np.argmax(prediction, axis=1)[0]
        # return str(prediction)
        return {"prediction":prediction.tolist()}
    else:
        return {"prediction":"Image doesnt exist in the dataset"}
