import os
import json
import re
import logging
from datetime import date
from typing import TypedDict, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId, SON
from pymongo import MongoClient
import requests
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and validate environment variables
def get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise EnvironmentError(f"Missing {var_name} in environment")
    logger.info(f"Loaded {var_name}")
    return value

# Load environment variables with validation
try:
    logger.info("Loading environment variables...")
    MONGODB_URI = get_required_env("MONGODB_URI")
    GOOGLE_API_KEY = get_required_env("GOOGLE_API_KEY")
    MALE_BN_API_URL = get_required_env("MALE_BN_API_URL")
    FEMALE_BN_API_URL = get_required_env("FEMALE_BN_API_URL")
except EnvironmentError as e:
    logger.error(f"Environment configuration error: {str(e)}")
    raise

# Initialize MongoDB client
try:
    logger.info("Initializing MongoDB connection...")
    tmp_client = MongoClient(MONGODB_URI)
    db = tmp_client.get_default_database()
    patients_col = db["patients"]
    metrics_col = db["healthmetrics"]
    medications_col = db["medications"]
    medicines_col = db["medicines"]
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Initialize LLM
try:
    logger.info("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY)
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Pydantic models
class Medication(BaseModel):
    medicationName: str
    dosage: str
    frequency: Optional[str] = None

class Medicine(BaseModel):
    name: str
    specialization: str
    description: Optional[str] = None

class Recommendations(BaseModel):
    patient_recommendations: Optional[List[str]] = None
    diet_plan: Optional[dict] = None
    exercise_plan: Optional[dict] = None
    nutrition_targets: Optional[dict] = None
    doctor_recommendations: Optional[List[str]] = None

class State(TypedDict):
    patient_data: dict
    sent_for: int
    risk_probabilities: dict
    recommendations: Recommendations
    selected_patient_recommendations: List[str]
    current_medications: List[Medication]
    available_medicines: List[Medicine]

# Helper functions
def parse_probability(prob_str: str) -> float:
    try:
        return float(prob_str.strip('%')) / 100
    except:
        return 0.0

def get_risk_probabilities(patient_data: dict) -> dict:
    payload = patient_data.copy()
    payload.pop('gender', None)
    gender = patient_data.get('gender', 'M')
    
    if gender == 'M':
        api_url = MALE_BN_API_URL
    elif gender == 'F':
        api_url = FEMALE_BN_API_URL
    else:
        raise ValueError("Invalid gender in patient data; must be 'M' or 'F'")

    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"BN API request failed: {str(e)}")
        return {
            "Health Risk Probabilities": {
                "Diabetes": "0%",
                "Heart Disease": "0%"
            }
        }

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
    try:
        o = orig['Health Risk Probabilities']
        n = new['Health Risk Probabilities']
        o_d = parse_probability(o['Diabetes'])
        o_c = parse_probability(o['Heart Disease'])
        n_d = parse_probability(n['Diabetes'])
        n_c = parse_probability(n['Heart Disease'])
        return ((n_d < o_d - 0.05 and n_c <= o_c + 0.01) or
                (n_c < o_c - 0.05 and n_d <= o_d + 0.01))
    except:
        return False

def get_patient_medications(patient_id: str) -> List[Medication]:
    try:
        medications = list(medications_col.find({"patientId": patient_id}))
        return [
            Medication(
                medicationName=med.get('medicationName', 'Unknown'),
                dosage=med.get('dosage', 'Unknown'),
                frequency=med.get('frequency')
            ) for med in medications
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medications: {str(e)}")
        return []

def get_available_medicines() -> List[Medicine]:
    try:
        medicines = list(medicines_col.find({}))
        return [
            Medicine(
                name=med.get('name', 'Unknown'),
                specialization=med.get('specialization', 'General'),
                description=med.get('description', '')
            ) for med in medicines
        ]
    except Exception as e:
        logger.error(f"Failed to fetch medicines database: {str(e)}")
        return []

# Graph nodes
def risk_assessment(state: State) -> dict:
    try:
        probs = get_risk_probabilities(state['patient_data'])
        return {'risk_probabilities': probs}
    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}")
        return {
            'risk_probabilities': {
                "Health Risk Probabilities": {
                    "Diabetes": "0%",
                    "Heart Disease": "0%"
                }
            }
        }

def generate_recommendations(state: State) -> dict:
    try:
        pd = state['patient_data']
        probs = state['risk_probabilities']['Health Risk Probabilities']
        sent_for = state['sent_for']
        medications = state.get('current_medications', [])
        available_meds = state.get('available_medicines', [])

        # Filter medicines by specialization
        relevant_meds = []
        if sent_for == 1:  # Cardiology
            relevant_meds = [m for m in available_meds if 'cardiology' in m.specialization.lower()]
        elif sent_for == 2:  # Endocrinology
            relevant_meds = [m for m in available_meds if 'endocrinology' in m.specialization.lower()]

        meds_info = []
        for med in relevant_meds:
            info = f"- {med.name}"
            if med.description:
                info += f" (description: {med.description})"
            meds_info.append(info)

        # Safely get all patient data with defaults
        patient_info = {
            'bp': pd.get('Blood_Pressure', 'N/A'),
            'bmi': pd.get('BMI', 'N/A'),
            'glucose': pd.get('glucose', 'N/A'),
            'cvd_risk': probs.get('Heart Disease', 'N/A'),
            'diabetes_risk': probs.get('Diabetes', 'N/A'),
            'comorbidities': "Prediabetes" if parse_probability(probs.get('Diabetes', '0%')) > 0.25 else "None noted",
            'exercise': f"{pd.get('Exercise_Hours_Per_Week', 0)} hrs/week",
            'diet': pd.get('Diet', 'Unknown'),
            'smoking_status': "Smoker" if pd.get('is_smoking', False) else "Non-smoker",
            'medications_list': "\n- ".join([f"{m.medicationName} {m.dosage}" + (f" ({m.frequency})" if m.frequency else "") for m in medications]),
            'medications_count': len(medications),
            'medications': ", ".join([f"{m.medicationName}" for m in medications]),
            'available_meds': "\n".join(meds_info) if meds_info else "No specific medications in database"
        }

        if sent_for == 0:
            instruction = (
                "Provide up to five lifestyle recommendations in 'patient_recommendations'.\n"
                "Include a diet plan in 'diet_plan' with description, calories, and meals.\n"
                "Include an exercise plan in 'exercise_plan' with type, duration, frequency.\n"
                "Include nutrition targets in 'nutrition_targets'.\n"
                "Set 'doctor_recommendations' to null.\n"
                "Current medications: {medications}"
            ).format(medications=patient_info['medications'])
        
        elif sent_for == 1:
            instruction = (
                "Provide cardiology recommendations in 'doctor_recommendations'.\n"
                "Include risk stratification, diagnostics, medications, and follow-up.\n"
                "Patient Data:\n"
                "- Vitals: {bp}, BMI {bmi}\n"
                "- Scores: ASCVD {cvd_risk}, Diabetes {diabetes_risk}\n"
                "- Current Meds: {medications_count} drugs\n"
                "- Resources: {available_meds}"
            ).format(**patient_info)
        
        elif sent_for == 2:
            instruction = (
                "Provide endocrinology recommendations in 'doctor_recommendations'.\n"
                "Include risk factors, diagnostics, medications, and monitoring.\n"
                "Patient Data:\n"
                "- Glucose: {glucose}, BMI: {bmi}\n"
                "- Current Meds: {medications_count} drugs\n"
                "- Resources: {available_meds}"
            ).format(**patient_info)
        
        else:
            raise HTTPException(status_code=400, detail='Invalid sent_for value')

        prompt = (
            f"Patient Data: {pd}\n"
            f"Current Medications: {patient_info['medications_list']}\n"
            f"Diabetes Risk: {probs.get('Diabetes', 'N/A')}\n"
            f"CVD Risk: {probs.get('Heart Disease', 'N/A')}\n\n"
            f"{instruction}\n"
            f"Return only valid JSON."
        )
        
        response = llm.invoke(prompt)
        json_str = re.search(r'\{.*\}', response.content, re.DOTALL).group(0)
        json_data = json.loads(json_str)
        
        # Ensure required fields exist
        if sent_for == 0:
            if 'patient_recommendations' not in json_data:
                json_data['patient_recommendations'] = []
            if 'diet_plan' not in json_data:
                json_data['diet_plan'] = {"description": "", "calories": 0, "meals": []}
            if 'exercise_plan' not in json_data:
                json_data['exercise_plan'] = {"type": "", "duration": 0, "frequency": 0}
            if 'nutrition_targets' not in json_data:
                json_data['nutrition_targets'] = {}
            json_data['doctor_recommendations'] = None
        else:
            if 'doctor_recommendations' not in json_data:
                json_data['doctor_recommendations'] = []
            json_data['patient_recommendations'] = None
            json_data['diet_plan'] = None
            json_data['exercise_plan'] = None
            json_data['nutrition_targets'] = None
            
        recs = Recommendations(**json_data)
        return {'recommendations': recs}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {str(e)}")
        return {
            'recommendations': Recommendations(
                patient_recommendations=["Error generating recommendations"],
                doctor_recommendations=["Error generating recommendations"]
            )
        }
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return {
            'recommendations': Recommendations(
                patient_recommendations=["Error generating recommendations"],
                doctor_recommendations=["Error generating recommendations"]
            )
        }

def evaluate_recommendations(state: State) -> dict:
    if state['sent_for'] != 0:
        return {'selected_patient_recommendations': []}
    
    try:
        original = state['risk_probabilities']
        selected = []
        for rec in state['recommendations'].patient_recommendations or []:
            kind = classify_recommendation(rec)
            if kind != 'Other':
                adj = adjust_metrics(state['patient_data'], kind)
                new_probs = get_risk_probabilities(adj)
                if is_effective(original, new_probs):
                    selected.append(rec)
        return {'selected_patient_recommendations': selected[:3]}  # Return max 3 recommendations
    except:
        return {'selected_patient_recommendations': []}

def output_results(state: State) -> dict:
    try:
        probs = state['risk_probabilities']['Health Risk Probabilities']
        result = {
            'diabetes_probability': probs.get('Diabetes', '0%'),
            'cvd_probability': probs.get('Heart Disease', '0%'),
            'current_medications': [{
                'medicationName': m.medicationName,
                'dosage': m.dosage,
                'frequency': m.frequency
            } for m in state.get('current_medications', [])]
        }
        
        if state['sent_for'] == 0:
            result.update({
                'patient_recommendations': state.get('selected_patient_recommendations', []),
                'diet_plan': state['recommendations'].diet_plan or {},
                'exercise_plan': state['recommendations'].exercise_plan or {},
                'nutrition_targets': state['recommendations'].nutrition_targets or {}
            })
        else:
            result['doctor_recommendations'] = state['recommendations'].doctor_recommendations or []
        
        return result
    except:
        return {
            'error': 'Failed to generate results',
            'diabetes_probability': '0%',
            'cvd_probability': '0%',
            'current_medications': []
        }

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
    except:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")

    try:
        patient = patients_col.find_one({"_id": oid})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        metrics = list(metrics_col.find({"patientId": patient_id}).sort([('createdAt', -1)]).limit(1))
        if metrics:
            patient.update(metrics[0])

        # Get patient data with defaults
        patient_data = {
            "Blood_Pressure": patient.get('bloodPressure', 0),
            "Age": patient.get('anchorAge', 30),
            "Exercise_Hours_Per_Week": patient.get('exerciseHoursPerWeek', 0),
            "Diet": patient.get('diet', 'Unknown'),
            "Sleep_Hours_Per_Day": patient.get('sleepHoursPerDay', 7),
            "Stress_Level": patient.get('stressLevel', 0),
            "glucose": patient.get('glucose', 0),
            "BMI": patient.get('bmi', 0),
            "hypertension": 1 if patient.get("bloodPressure", 0) > 130 else 0,
            "is_smoking": patient.get('isSmoker', False),
            "hemoglobin_a1c": patient.get('hemoglobinA1c', 0),
            "Diabetes_pedigree": patient.get('diabetesPedigree', 0),
            "CVD_Family_History": patient.get('ckdFamilyHistory', 0),
            "ld_value": patient.get('cholesterolLDL', 0),
            "admission_tsh": patient.get('admissionSOH', 0),
            "is_alcohol_user": patient.get('isAlcoholUser', False),
            "creatine_kinase_ck": patient.get('creatineKinaseCK', 0),
            "gender": 'M' if str(patient.get('gender', 'M')).lower().startswith('m') else 'F',
        }

        initial_state = {
            'patient_data': patient_data,
            'sent_for': sent_for,
            'current_medications': get_patient_medications(patient_id),
            'available_medicines': get_available_medicines()
        }
        
        result = await graph.ainvoke(initial_state)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
