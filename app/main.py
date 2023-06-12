import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
from predict import predict_model

MODEL_PATH = "reduced_model.h5"
IMAGE_SIZE = (256, 256, 3)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/process')
async def process(request: Request, file: UploadFile = File(...)):
    file_location = f"{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read()) #to get file locally

    img = np.asarray(Image.open(file_location).resize((IMAGE_SIZE[0], IMAGE_SIZE[1])), dtype="float32").reshape(IMAGE_SIZE)
    normalized_img = img/255
    predicted_mask = predict_model(MODEL_PATH, normalized_img.reshape((1,)+IMAGE_SIZE)).reshape((IMAGE_SIZE[0], IMAGE_SIZE[1]))*255

    predicted_mask_img = f"{file_location.split('.')[0]}_mask.jpg"
    Image.fromarray(predicted_mask).convert("L").save(predicted_mask_img)
    Image.open(file_location).resize((IMAGE_SIZE[0], IMAGE_SIZE[1])).save(file_location)


    return templates.TemplateResponse("index.html", {"request": request, "file_location": file_location, "segmentation_mask": predicted_mask_img})

@app.get("/{filename}")
def download_segmentation(filename: str):
    print(filename)
    return FileResponse(filename, media_type="image/jpeg")
