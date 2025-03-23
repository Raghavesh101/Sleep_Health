from fastapi import FastAPI, HTTPException, Request, Response, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from wearable_app.app.models import UserData, UserPredictionData, GyroscopeData, PredictionData
from wearable_app.database.connect import db
from datetime import datetime
import joblib
import logging
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates and static files
templates = Jinja2Templates(directory="wearable_app/app/templates")

# Load the trained model and label encoders
model = joblib.load("wearable_app/app/sleep_disorder_model.pkl")
label_encoders = joblib.load("wearable_app/app/label_encoders.pkl")

# Threshold for wake events in gyroscope data
THRESHOLD_DISTANCE = 3

# Define the login page route
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Handle login form submission and authenticate with database
@app.post("/login")
async def login(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    user = db.users.find_one({"username": username, "password": password})
    if user:
        response.set_cookie(key="authenticated", value="true")
        response.set_cookie(key="username", value=username)
        return RedirectResponse(url="/", status_code=302)
    else:
        return HTMLResponse(content="Invalid credentials. Please try again.", status_code=401)

# Logout endpoint
@app.get("/logout")
async def logout(response: Response):
    response.delete_cookie("authenticated")
    response.delete_cookie("username")
    return RedirectResponse(url="/login")

# Homepage with authentication check
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    if request.cookies.get("authenticated") == "true":
        return templates.TemplateResponse("user_interface.html", {"request": request})
    return RedirectResponse(url="/login")

# Endpoint to save gyroscope data
@app.post("/save-gyroscope-data")
async def save_gyroscope_data(gyroscope_data: GyroscopeData):
    data_dict = gyroscope_data.dict()
    if data_dict["timestamp"] is None:
        data_dict["timestamp"] = datetime.now()

    result = db.gyroscope_data.insert_one(data_dict)
    if result.inserted_id:
        return {"message": "Gyroscope data saved successfully", "id": str(result.inserted_id)}
    else:
        raise HTTPException(status_code=500, detail="Failed to save gyroscope data")

# Endpoint to calculate sleep quality
@app.get("/calculate-sleep-quality")
async def calculate_sleep_quality():
    records = db.gyroscope_data.find({}, {"distance": 1})
    wake_count = sum(1 for record in records if record.get("distance", 0) > THRESHOLD_DISTANCE)

    if wake_count > 5:
        sleep_quality = "Very Poor"
    elif wake_count > 3:
        sleep_quality = "Poor"
    elif 2 <= wake_count <= 3:
        sleep_quality = "Good"
    else:
        sleep_quality = "Excellent"

    db.sleep_history.insert_one({
        "timestamp": datetime.now(),
        "sleep_quality": sleep_quality,
        "wake_count": wake_count
    })

    return {
        "wake_count": wake_count,
        "sleep_quality": sleep_quality
    }

# Endpoint to predict sleep disorder
@app.post("/predict-sleep-disorder")
async def predict_sleep_disorder(data: UserPredictionData):
    try:
        input_data = [
            label_encoders['Gender'].transform([data.gender])[0],
            data.age,
            label_encoders['Occupation'].transform([data.occupation])[0],
            data.sleep_duration,
            data.physical_activity,
            data.bmi,
            data.daily_steps,
            data.stress_level,
            data.blood_pressure
        ]

        prediction = model.predict([input_data])[0]

        disorder_mapping = {
            0: "Insomnia",
            1: "None",
            2: "Sleep Apnea"
        }
        disorder = disorder_mapping.get(prediction, "Unknown")

        return {"predicted_sleep_disorder": disorder}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Endpoint to save user data
@app.post("/save-user-data")
async def save_user_data(data: UserData):
    try:
        db.user_data.insert_one({**data.dict(), "timestamp": datetime.utcnow()})
        return {"message": "User data saved successfully"}
    except Exception as e:
        logging.error(f"Failed to save user data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save user data")

# Serve the blogs page
@app.get("/blogs", response_class=HTMLResponse)
async def blogs(request: Request):
    return templates.TemplateResponse("blogs.html", {"request": request})

# Endpoint to save predictions with sleep disorder, quality, and duration
@app.post("/save-predictions")
async def save_predictions(prediction_data: PredictionData):
    try:
        db.predictions.insert_one(prediction_data.dict())
        return JSONResponse(status_code=200, content={"message": "Predictions saved successfully"})
    except Exception as e:
        logging.error(f"Error saving predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to save predictions")

# Serve the insights page
@app.get("/insights", response_class=HTMLResponse)
async def insights(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request})

# Endpoint to get the last 7 days of insights data
@app.get("/get-insights-data")
async def get_insights_data():
    try:
        insights_data = list(
            db.predictions.find({}, {"timestamp": 1, "sleep_duration": 1, "sleep_quality": 1})
            .sort("timestamp", -1)
            .limit(7)
        )

        insights_data.reverse()

        result = [
            {
                "timestamp": entry["timestamp"].isoformat(),
                "sleep_duration": entry.get("sleep_duration", "N/A"),
                "sleep_quality": entry.get("sleep_quality", "N/A")
            }
            for entry in insights_data
        ]

        return JSONResponse(content=result)

    except Exception as e:
        logging.error(f"Error fetching insights data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch insights data")

# Serve the recommendations page
@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations(request: Request):
    return templates.TemplateResponse("recommendations.html", {"request": request})

# Chatbot endpoint for recommendations
@app.post("/recommendations-chat")
async def recommendations_chat(request: Request):
    try:
        user_query = await request.json()
        query_text = user_query.get("query", "").lower()

        recent_data = list(db.predictions.find().sort("timestamp", -1).limit(7))
        
        if not recent_data:
            response = "I'm sorry, but I couldn't find any recent data to base recommendations on."
        else:
            sleep_summary = "\n".join([
                f"Date: {entry['timestamp'].strftime('%Y-%m-%d')}, "
                f"Sleep Duration: {entry['sleep_duration']} hours, "
                f"Quality: {entry['sleep_quality']}, "
                f"Disorder: {entry['sleep_disorder']}" 
                for entry in recent_data
            ])

            prompt = f"""
            You are a health assistant that provides concise and professional sleep recommendations. Based on the user's recent sleep data below, generate a response in clear, complete sentences with no more than 50 words, formatted as bullet points.

            User's recent sleep data:
            {sleep_summary}

            User's question: "{query_text}"

            Please keep the response short, professional, and directly actionable.
            """

            openai_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in sleep health."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )

            response = openai_response.choices[0].message['content'].strip()

        return JSONResponse({"response": response})

    except Exception as e:
        logging.error(f"Error in recommendations_chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chatbot query.")
