import os 
import asyncio
import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pyngrok import ngrok
from typing import List, Optional

from ml_functions.model_container import ModelContainer
from ml_functions.text_extraction_utils import pytesseract_get_text_from_image
from ml_functions.gemini_helper import gemini_parse_text_to_json
from dotenv import load_dotenv

load_dotenv()


nest_asyncio.apply()
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN")) 


# --- Model and prediction logic ---

model_container = ModelContainer(model_path=os.getenv("MODEL_PATH"))


app = FastAPI(
    title="Image Screen Detector API",
    description="A simple API to detect if an image was taken from a screen."
)

class PredictionResult(BaseModel):
    filename: str
    probability: float
    extracted_text: Optional[str] = None

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]

# Template directory setup
templates = Jinja2Templates(directory="templates")

# Endpoint for serving the HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict_images(files: List[UploadFile] = File(...)):
    """
    Accepts a list of image files and returns the probability that each one was taken from a screen,
    along with any extracted text.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    decoded_images = []
    filenames = []
    
    try:
        # Step 1: Decode and collect all images and filenames
        for file in files:
            contents = await file.read()
            image = model_container.decode_image(contents)
            decoded_images.append(image)
            filenames.append(file.filename)
            
        # Step 2: Run batch inference to get probabilities efficiently
        probabilities = model_container.run_inference(decoded_images)
        
        # Step 3: Extract text from each image individually
        extracted_texts = [pytesseract_get_text_from_image(img) for img in decoded_images]
        formatted_receipts = [gemini_parse_text_to_json(text) for text in extracted_texts]
        print(formatted_receipts)
        # Step 4: Combine all results into a single list
        predictions = []
        for i, filename in enumerate(filenames):
            predictions.append({
                "filename": filename,
                "probability": probabilities[i],
                "extracted_text": extracted_texts[i],
                "formatted_receipt": formatted_receipts[i]
            })
        
        return {"predictions": predictions}

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


async def run_server_and_ngrok():
    # public_url = ngrok.connect(8000)
    # print(f"\nPublic URL: {public_url}\n")
    config = uvicorn.Config(app, host="0.0.00", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_server_and_ngrok())