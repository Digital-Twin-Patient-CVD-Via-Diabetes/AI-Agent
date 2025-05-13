import os
import json
import re
from datetime import date
from typing import TypedDict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId, SON
from pymongo import MongoClient
import requests
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

# Load environment variables directly from system environment
MONGODB_URI = os.getenv("MONGODB_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MALE_BN_API_URL = os.getenv("MALE_BN_API_URL")
FEMALE_BN_API_URL = os.getenv("FEMALE_BN_API_URL")

# Validate required environment variables
if not MONGODB_URI:
    raise EnvironmentError("Missing MONGODB_URI in environment")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY in environment")
if not MALE_BN_API_URL or not FEMALE_BN_API_URL:
    raise EnvironmentError("Missing gender-specific BN API URLs in environment")

# Initialize MongoDB client
tmp_client = MongoClient(MONGODB_URI)
db = tmp_client.get_default_database()
patients_col = db["patients"]
metrics_col = db["healthmetrics"]

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY)

# Pydantic models
class Recommendations(BaseModel):
    patient_recommendations: Optional[list[str]] = None
    diet_plan: Optional[dict] = None
    exercise_plan: Optional[dict] = None
    nutrition_targets: Optional[dict] = None
    doctor_recommendations: Optional[list[str]] = None

class State(TypedDict):
    patient_data: dict
    sent_for: int
    risk_probabilities: dict
    recommendations: Recommendations
    selected_patient_recommendations: list[str]

# Helper functions
def parse_probability(prob_str: str) -> float:
    return float(prob_str.strip('%')) / 100

def get_risk_probabilities(patient_data: dict) -> dict:
    payload = patient_data.copy()
    payload.pop('gender', None)
    gender = patient_data.get('gender')
    
    if gender == 'M':
        api_url = MALE_BN_API_URL
    elif gender == 'F':
        api_url = FEMALE_BN_API_URL
    else:
        raise ValueError("Invalid gender in patient data; must be 'M' or 'F'")

    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"BN service error: {response.status_code}")
    return response.json()

def classify_recommendation(text: str) -> str:
    t = text.lower()
    if 'exercise' in t:
        return 'Physical Activity'
    if 'diet' in t or 'nutrition' in t:
        return 'Diet'
    if 'smoking' in t:
        return 'Smoking Cessation'
    return 'Other'

def adjust_metrics(data: dict, kind: str) -> dict:
    d = data.copy()
    if kind == 'Physical Activity':
        d['Exercise_Hours_Per_Week'] = d.get('Exercise_Hours_Per_Week', 0) + 2
    if kind == 'Diet':
        if 'BMI' in d:
            d['BMI'] = max(d['BMI'] - 1, 0)
        if 'glucose' in d:
            d['glucose'] = max(d['glucose'] - 10, 0)
    if kind == 'Smoking Cessation':
        d['is_smoking'] = False
    return d

def is_effective(orig: dict, new: dict) -> bool:
    o = orig['Health Risk Probabilities']
    n = new['Health Risk Probabilities']
    o_d = parse_probability(o['Diabetes'])
    o_c = parse_probability(o['Heart Disease'])
    n_d = parse_probability(n['Diabetes'])
    n_c = parse_probability(n['Heart Disease'])
    return ((n_d < o_d - 0.05 and n_c <= o_c + 0.01) or
            (n_c < o_c - 0.05 and n_d <= o_d + 0.01))

# Graph nodes
def risk_assessment(state: State) -> dict:
    probs = get_risk_probabilities(state['patient_data'])
    return {'risk_probabilities': probs}

def generate_recommendations(state: State) -> dict:
    pd = state['patient_data']
    probs = state['risk_probabilities']['Health Risk Probabilities']
    sent_for = state['sent_for']

    if sent_for == 0:
        instruction = (
            "Provide up to five lifestyle and behavior change recommendations in 'patient_recommendations'.\n"
            "Additionally, you MUST provide a diet plan tailored for Egyptian patients in 'diet_plan', which must be a dictionary with 'description' (string describing the diet, including Egyptian foods), 'calories' (integer, daily calorie target), and 'meals' (list of strings, example meals).\n"
            "You MUST provide an exercise plan in 'exercise_plan', which must be a dictionary with 'type' (string, e.g., 'aerobic'), 'duration' (integer, minutes per session), 'frequency' (integer, sessions per week).\n"
            "You MUST provide nutrition targets in 'nutrition_targets', which must be a dictionary with target values for relevant metrics, e.g., 'target_BMI', 'target_glucose', etc.\n"
            "Set 'doctor_recommendations' to null.\n"
            "**Critical Instruction:** Do NOT omit 'diet_plan', 'exercise_plan', or 'nutrition_targets'. These fields are required and must be populated with appropriate values based on the patient data.\n"
            "Here's an example of the expected JSON output:\n"
            "{\n"
            "  \"patient_recommendations\": [\"Increase water intake\", \"Reduce sugar consumption\"],\n"
            "  \"diet_plan\": {\"description\": \"A balanced diet with Egyptian staples like ful medames and koshari\", \"calories\": 2000, \"meals\": [\"Ful medames with bread\", \"Grilled chicken with rice\"]},\n"
            "  \"exercise_plan\": {\"type\": \"aerobic\", \"duration\": 30, \"frequency\": 5},\n"
            "  \"nutrition_targets\": {\"target_BMI\": 25.0, \"target_glucose\": 100},\n"
            "  \"doctor_recommendations\": null\n"
            "}"
        )
    elif sent_for == 1:
        instruction = (
            "Provide up to three medical action recommendations for a cardiologist in 'doctor_recommendations'. "
            "Notify about comorbid conditions (e.g., prediabetes) and caution against medications that may worsen those conditions.\n"
            "Set 'patient_recommendations', 'diet_plan', 'exercise_plan', 'nutrition_targets' to null."
        )
    elif sent_for == 2:
        instruction = (
            "Provide up to three medical action recommendations for an endocrinologist in 'doctor_recommendations'. "
            "Notify about comorbid conditions (e.g., CVD risk) and caution against medications that may worsen those conditions.\n"
            "Set 'patient_recommendations', 'diet_plan', 'exercise_plan', 'nutrition_targets' to null."
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid sent_for value')

    prompt = (
        f"Based on the following patient profile and risk probabilities, generate recommendations.\n"
        f"Patient Data: {pd}\n"
        f"Diabetes Risk: {probs['Diabetes']}\n"
        f"CVD Risk: {probs['Heart Disease']}\n\n"
        f"{instruction}\n"
        f"Return only the JSON object, without any additional text or explanations."
    )
    response = llm.invoke(prompt)
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response.content
        recs_dict = json.loads(json_str)
        recs = Recommendations(**recs_dict)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from the language model response.")
    except Exception as e:
        raise ValueError(f"Error parsing recommendations: {str(e)}")

    if sent_for == 0:
        if not recs.diet_plan or not recs.exercise_plan or not recs.nutrition_targets:
            raise ValueError("Diet plan, exercise plan, or nutrition targets are missing in the recommendations.")

    return {'recommendations': recs}

def evaluate_recommendations(state: State) -> dict:
    if state['sent_for'] != 0:
        return {'selected_patient_recommendations': []}
    if state['recommendations'] is None:
        raise ValueError("Recommendations are None for sent_for == 0")
    original = state['risk_probabilities']
    selected = []
    for rec in state['recommendations'].patient_recommendations or []:
        kind = classify_recommendation(rec)
        if kind != 'Other':
            adj = adjust_metrics(state['patient_data'], kind)
            new_probs = get_risk_probabilities(adj)
            if is_effective(original, new_probs):
                selected.append(rec)
    return {'selected_patient_recommendations': selected}

def output_results(state: State) -> dict:
    probs = state['risk_probabilities']['Health Risk Probabilities']
    sent_for = state['sent_for']
    result = {
        'diabetes_probability': probs['Diabetes'],
        'cvd_probability': probs['Heart Disease']
    }
    if sent_for == 0:
        result['patient_recommendations'] = state['selected_patient_recommendations'][:3]
        result['diet_plan'] = state['recommendations'].diet_plan
        result['exercise_plan'] = state['recommendations'].exercise_plan
        result['nutrition_targets'] = state['recommendations'].nutrition_targets
    elif sent_for == 1:
        result['doctor_recommendations'] = state['recommendations'].doctor_recommendations[:3]
    elif sent_for == 2:
        result['doctor_recommendations'] = state['recommendations'].doctor_recommendations[:3]
    return result

# Build and compile state graph
graph_builder = StateGraph(State)
for node in ['risk_assessment', 'generate_recommendations', 'evaluate_recommendations', 'output_results']:
    graph_builder.add_node(node, globals()[node])
graph_builder.add_edge(START, 'risk_assessment')
graph_builder.add_edge('risk_assessment', 'generate_recommendations')
graph_builder.add_edge('generate_recommendations', 'evaluate_recommendations')
graph_builder.add_edge('evaluate_recommendations', 'output_results')
graph_builder.add_edge('output_results', END)
graph = graph_builder.compile()

# FastAPI app
app = FastAPI()

@app.get("/recommendations/{patient_id}")
async def get_recommendations(patient_id: str, sent_for: Optional[int] = 0):
    try:
        oid = ObjectId(patient_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")

    patient = patients_col.find_one({"_id": oid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    metrics = list(
        metrics_col.find({"patientId": patient_id})
                   .sort([('createdAt', -1)])
                   .limit(1)
    )
    if metrics:
        patient.update(metrics[0])

    patient_data = {
        "Age": (date.today() - patient['birthDate'].date()).days // 365,
        "Blood_Pressure": patient.get('bloodPressure'),
        "BMI": patient.get('bmi'),
        "glucose": patient.get('glucose'),
        "Exercise_Hours_Per_Week": patient.get('exerciseHoursPerWeek'),
        "Diet": patient.get('diet'),
        "Sleep_Hours_Per_Day": patient.get('sleepHoursPerDay'),
        "Stress_Level": patient.get('stressLevel'),
        "is_smoking": patient.get('isSmoker'),
        "is_alcohol_user": patient.get('isAlcoholUser'),
        "diabetesPedigree": patient.get('diabetesPedigree'),
        "gender": 'M' if patient['gender'].lower().startswith('m') else 'F'
    }

    initial_state = {'patient_data': patient_data, 'sent_for': sent_for} 
    result = await graph.ainvoke(initial_state)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
