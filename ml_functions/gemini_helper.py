import json
import os
import requests

# --- API Configuration ---
# Set the API key from an environment variable for security
gemini_api_key = os.getenv("GEMINI_API_KEY")

# The URL for the Gemini API, using the specified model and API key
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={gemini_api_key}"


def gemini_parse_text_to_json(ocr_text: str) -> dict:
    if not ocr_text:
        return {"amount": "-", "store": "-", "date": "-", "items": []}

    prompt = f"""
    Analyze the following receipt text and extract the total amount, the store name, and the date.
    Also, identify the items purchased, their quantity, and their price.
    If any information is missing or cannot be identified with certainty, use a hyphen "-" for text fields and an empty array [] for the items array. DO NOT create or hallucinate any data.
    Return ONLY a JSON dictionary with the following keys:
    'amount': The total amount of the purchase.
    'store': The name of the store.
    'date': The date of the transaction.
    'items': An array of objects, where each object represents an item with keys 'name', 'quantity', and 'price'.
    If no items are found, the 'items' array should be empty [].

    Receipt text:
    {ocr_text}
    """
    
    # Construct the payload for the API request
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
        
        # Extract the text content from the API response
        response_data = response.json()
        response_text = response_data['candidates'][0]['content']['parts'][0]['text'].strip()

        # Robust JSON parsing: remove markdown blocks and fix common errors
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1).strip()
        if response_text.endswith("```"):
            response_text = response_text.replace("```", "", 1).strip()
        if response_text.endswith(','):
            response_text = response_text.rstrip(',') + '}'
        
        extracted_data = json.loads(response_text)
        return extracted_data

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return {"amount": "-", "store": "-", "date": "-", "items": []}
    except (KeyError, IndexError) as e:
        print(f"API Response Parsing Error: Could not find expected data in response. Details: {e}")
        return {"amount": "-", "store": "-", "date": "-", "items": []}
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON. Original error: {e}")
        return {"amount": "-", "store": "-", "date": "-", "items": []}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"amount": "-", "store": "-", "date": "-", "items": []}

# --- Example Usage ---
if __name__ == "__main__":
    sample_text = """
    АТБ-Маркет
    Касовий чек № 1234
    Чек від 28.11.2023 10:45
    
    1. Молоко АТБ 2.5% 1 шт. x 35.50 = 35.50
    2. Хліб Український 1 шт. x 20.00 = 20.00
    3. Сир Гауда 0.250 кг x 150.00 = 37.50
    
    Всього: 93.00
    """
    
    analysis_result = gemini_parse_text_to_json(sample_text)
    print(json.dumps(analysis_result, ensure_ascii=False, indent=4))