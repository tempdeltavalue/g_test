import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn 

IMG_FOLDER = 'annotation_tool/data/receipts_photos'
RESULTS_BASE_PATH = 'annotation_tool/data'

app = FastAPI()

app.mount("/images", StaticFiles(directory=IMG_FOLDER), name="images")

templates = Jinja2Templates(directory="annotation_tool/templates")

def load_data():
    all_data = []
    for filename in ['tess_screen_photos.json', 'tess_normal_photos.json']:
        try:
            filepath = f"{RESULTS_BASE_PATH}/{filename}"
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        except FileNotFoundError:
            print(f"File '{filepath}' not found.")
    return all_data

data = load_data()
current_index = 0

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    global current_index
    if not data:
        return templates.TemplateResponse("ann_tool_index.html", {"request": request, "error": "No data loaded."})

    entry = data[current_index]
    image_id = entry.get('image_id', '')
    ocr_text = entry.get('ocr_text', 'No OCR text found.')

    has_prev = current_index > 0
    has_next = current_index < len(data) - 1

    return templates.TemplateResponse(
        "ann_tool_index.html",
        {
            "request": request,
            "image_id": image_id,
            "ocr_text": ocr_text,
            "current_index": current_index + 1,
            "total_count": len(data),
            "has_prev": has_prev,
            "has_next": has_next
        }
    )

@app.get("/next")
async def next_entry():
    global current_index
    if current_index < len(data) - 1:
        current_index += 1
    return HTMLResponse(content="Redirecting...", status_code=302, headers={"Location": "/"})

@app.get("/prev")
async def prev_entry():
    global current_index
    if current_index > 0:
        current_index -= 1
    return HTMLResponse(content="Redirecting...", status_code=302, headers={"Location": "/"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
