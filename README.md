# HOUSE Prediction API

An end-to-end machine learning web application built with FastAPI for predicting house prices based on user-provided property features. The project includes a trained regression model, a REST API, an interactive web interface, and optional AI-powered explanations using a large language model.

## Overview

This project uses the California Housing dataset and a RandomForestRegressor model to estimate house prices. The app provides:

- A FastAPI backend for price prediction
- A simple web UI for entering house details and viewing predictions
- CSV upload support for batch predictions
- AI-generated market analysis and explanations via Groq/Llama
- Health and status endpoints for quick checks

## Features

- Predict house price from 8 input features:
  - Median income
  - House age
  - Average rooms
  - Average bedrooms
  - Population
  - Average occupancy
  - Latitude
  - Longitude
- Return predicted price, confidence range, and short summary
- Provide AI-based explanation and market analysis through the LLM integration
- Upload a CSV file containing the required input columns and download predictions

## Tech Stack

- Python
- FastAPI
- Uvicorn
- scikit-learn
- pandas
- joblib
- python-dotenv
- requests
- HTML/JavaScript frontend

## Project Structure

- main.py: FastAPI app, prediction routes, health check, and CSV prediction endpoint
- train.py: trains the model and saves model artifacts
- llm_service.py: LLM explanation helper
- explore.py: simple data exploration script
- test_llm.py: small test for the LLM service
- index.html: frontend for the web app
- requirements.txt: Python dependencies

## Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a .env file in the project root with your LLM API credentials:

```env
GROQ_API_KEY=your_api_key_here
```

Optional variables:

```env
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_MODEL=llama-3.3-70b-versatile
```

## Train the Model

Run the training script to build and save the model files:

```bash
python train.py
```

This will generate:

- house_model.joblib
- house_features.joblib

## Run the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Then open:

- http://127.0.0.1:8000/ for the web UI
- http://127.0.0.1:8000/docs for the Swagger documentation

## API Endpoints

### GET /
Returns the web interface.

### GET /health
Returns the server status and model information.

### POST /predict
Predicts the price for a single house input.

Example body:

```json
{
  "MedInc": 3.5,
  "HouseAge": 20,
  "AveRooms": 5.2,
  "AveBedrms": 1.0,
  "Population": 1200,
  "AveOccup": 2.5,
  "Latitude": 37.77,
  "Longitude": -122.42
}
```

### POST /predict-with-chat
Predicts the price and returns AI-generated analysis.

### POST /chat
Sends a prompt to the LLM and returns the reply.

### POST /predict-file
Uploads a CSV file with the required columns and returns a CSV file with predictions.

## Notes

- The model is trained on the California Housing dataset.
- The prediction output uses a scaled format where the model predicts a value in hundreds of thousands of dollars.
- The AI explanation feature requires a valid Groq API key.

## Example Usage

You can test the API with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.5,
    "HouseAge": 20,
    "AveRooms": 5.2,
    "AveBedrms": 1.0,
    "Population": 1200,
    "AveOccup": 2.5,
    "Latitude": 37.77,
    "Longitude": -122.42
  }'
```

