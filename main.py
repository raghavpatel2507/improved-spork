from fastapi import FastAPI,File,UploadFile
import uvicorn
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
app=FastAPI()

MODEL=tf.keras.models.load_model('potatoes.h5')
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]

@app.get('/ping')
async def ping():
    return 'hello'

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
        file:UploadFile =File(...)


):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions=MODEL.predict(img_batch)
    try:
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    except IndexError:
        predicted_class = "healthy"
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)