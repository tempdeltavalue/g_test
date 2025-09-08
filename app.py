import os
import asyncio
import nest_asyncio
import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pyngrok import ngrok
from typing import List, Optional, Dict

# Import the get_db_session function and the database model
from database.db_utils import get_db_session, ReceiptResult

from ml_functions.model_container import ModelContainer
from ml_functions.text_extraction_utils import pytesseract_get_text_from_image
from ml_functions.gemini_helper import gemini_parse_text_to_json
from dotenv import load_dotenv

load_dotenv()

nest_asyncio.apply()
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN")) 

# --- Model and prediction logic ---
model_container = ModelContainer(model_type=os.getenv("MODEL_TYPE"), 
                                 model_path=os.getenv("MODEL_PATH"))

app = FastAPI(
    title="Image Screen Detector API",
    description="A simple API to detect if an image was taken from a screen."
)

class PredictionResult(BaseModel):
    filename: str
    probability_class_1: float
    probability_class_0: float
    predicted_class: int
    confidence: float
    extracted_text: Optional[str] = None
    formatted_receipt: Optional[str] = None

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]

# Template directory setup
templates = Jinja2Templates(directory="templates")

# Endpoint for serving the HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict_images(
    files: List[UploadFile] = File(...), 
    use_gemini_api: bool = Form(False)
):
    """
    Accepts a list of image files, processes them, and saves the results to the database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    decoded_images = []
    filenames = []
    
    # Get a new database session
    db_session = get_db_session()
    
    try:
        # Step 1: Decode and collect all images and filenames
        for file in files:
            contents = await file.read()
            image = model_container.decode_image(contents)
            decoded_images.append(image)
            filenames.append(file.filename)
            
        # Step 2: Run batch inference
        probabilities = model_container.run_inference(decoded_images)
        
        # Step 3: Extract text
        extracted_texts = [pytesseract_get_text_from_image(img) for img in decoded_images]
        
        # Step 4: Process with Gemini API
        if use_gemini_api:
            formatted_receipts = [json.dumps(gemini_parse_text_to_json(text)) for text in extracted_texts]
        else:
            formatted_receipts = [None] * len(extracted_texts)

        # Step 5: Combine results and save to database
        predictions = []
        for i, filename in enumerate(filenames):
            prob_class1 = float(probabilities[i])
            prob_class0 = 1 - prob_class1
            predicted_class = 1 if prob_class1 >= 0.5 else 0
            confidence = max(prob_class1, prob_class0)

            result_dict = {
                "filename": filename,
                "probability_class_1": prob_class1,
                "probability_class_0": prob_class0,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "extracted_text": extracted_texts[i],
                "formatted_receipt": formatted_receipts[i]
            }
            predictions.append(result_dict)
            
            # Create a database object and add to the session
            receipt_entry = ReceiptResult(**result_dict)
            db_session.add(receipt_entry)
        
        db_session.commit()
        
        # Return the response to the client
        return {"predictions": predictions}

    except Exception as e:
        db_session.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        db_session.close()

# --- Server setup ---
async def run_server_and_ngrok():
    public_url = ngrok.connect(8000)
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_server_and_ngrok())