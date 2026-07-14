import io
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="California House Prediction API")

model = joblib.load(BASE_DIR / "house_model.joblib")
features = joblib.load(BASE_DIR / "house_features.joblib")

LLM_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of Neighbourhood")
    HouseAge: float = Field(ge=0,description="average age of houses")
    AveRooms: float = Field(gt=0,description="average no of Rooms")
    AveBedrms: float = Field(gt=0,description="average age of Bedrooms")
    Population: float = Field(gt=0,description="Total populations")
    AveOccup: float = Field(gt=0,description="average no of Occupation of peoples")
    Latitude: float = Field(ge=32,description="Latitude")
    Longitude: float = Field(ge=-125,description="Longitude")
    





# home

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")

def health():

    return{

        "status":"running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"

    }


#  prediction

@app.post("/predict")

def predict(house: HouseFeatures):

    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])


        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"{predicted:.2f} hundred thousands",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }



    except Exception as e:
        raise HTTPException(

            status_code= 500,
            detail = f"prediction failed:{str(e)}"
        )


class ChatRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_LLM_MODEL


class PredictWithChatRequest(BaseModel):
    house: HouseFeatures
    explanation_prompt: str = """Analyze this house price prediction and return a JSON object with the following structure (and nothing else, no markdown, pure JSON):
{
  "price_level": "High/Medium/Low (compared to market)",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "price_drivers": {"positive": ["driver 1", "driver 2"], "negative": ["driver 1"]},
  "location_impact": "Brief assessment of latitude/longitude impact",
  "investment_potential": "Good/Moderate/Risky",
  "investment_reason": "Why this is a good/moderate/risky investment",
  "market_comparison": "Is this overpriced or underpriced?",
  "recommendations": ["recommendation 1", "recommendation 2"]
}"""
    model: str = DEFAULT_LLM_MODEL


@app.post("/predict-with-chat")
def predict_with_chat(request: PredictWithChatRequest):
    try:
        input_data = pd.DataFrame([{
            "MedInc": request.house.MedInc,
            "HouseAge": request.house.HouseAge,
            "AveRooms": request.house.AveRooms,
            "AveBedrms": request.house.AveBedrms,
            "Population": request.house.Population,
            "AveOccup": request.house.AveOccup,
            "Latitude": request.house.Latitude,
            "Longitude": request.house.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000
        explanation_input = (
            f"Predicted house price: ${price_usd:,.0f} for a house with "
            f"MedInc={request.house.MedInc}, HouseAge={request.house.HouseAge}, "
            f"AveRooms={request.house.AveRooms}, AveBedrms={request.house.AveBedrms}, "
            f"Population={request.house.Population}, AveOccup={request.house.AveOccup}, "
            f"Latitude={request.house.Latitude}, Longitude={request.house.Longitude}. "
            f"{request.explanation_prompt}"
        )

        if not LLM_API_KEY:
            raise HTTPException(status_code=500, detail="LLM API key not configured")

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": explanation_input}],
            "temperature": 0.7
        }

        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=20)
        print(f"Groq Response Status: {response.status_code}")
        print(f"Groq Response: {response.text}")
        response.raise_for_status()
        data = response.json()
        explanation = data["choices"][0]["message"]["content"]

        # Try to parse JSON from LLM response
        try:
            # Extract JSON from response (in case there's extra text)
            json_str = explanation.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()
            llm_data = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            llm_data = {"analysis": explanation.strip()}

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"{predicted:.2f} hundred thousands",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}",
            "llm_analysis": llm_data
        }
    except requests.RequestException as e:
        print(f"Request error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected LLM response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failed: {str(e)}")


@app.post("/chat")
def chat(request: ChatRequest):
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="LLM API key not configured")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=20)
        print(f"Groq Response Status: {response.status_code}")
        print(f"Groq Response: {response.text}")
        response.raise_for_status()
        data = response.json()
        return {"reply": data["choices"][0]["message"]["content"]}
    except requests.RequestException as e:
        print(f"Request error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected LLM response: {str(e)}")


@app.post("/predict-file")

async def predict_file(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a csv file only"
        )

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"unable to parse CSV file: {str(e)}"
        )

    required_columns = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from file {missing_columns}"
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="the uploaded file has no data rows"
        )

    try:
        predictions = model.predict(df[required_columns])
        df["predicted_price"] = predictions * 100000
        df["predicted_price"] = df["predicted_price"].apply(lambda x: f"${x:,.0f}")
        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "content-Disposition": "attachment;filename=predictions.csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )
    

 

        