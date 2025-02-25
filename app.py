import streamlit as st
import requests
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from cryptography.fernet import Fernet
import sqlite3
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
from typing import List, Dict, Tuple

# Load environment variables
load_dotenv()
ICD10_API_KEY = st.secrets["ICD10_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
ENCRYPTION_KEY = st.secrets["ENCRYPTION_KEY"]

# Initialize core components
#client = Groq(api_key=GROQ_API_KEY)
cipher_suite = Fernet(ENCRYPTION_KEY.encode())
DATABASE_NAME = "secure_patient_data.db"

# Medical Normalization System
class MedicalNormalizer:
    def __init__(self):
        self.symptom_map = {
            # Pain terms
            "hrt pain": "chest pain",
            "stummy ache": "abdominal pain",
            "tummy hurt": "abdominal pain",
            "belly ache": "abdominal pain",
            
            # Respiratory
            "can't breathe": "shortness of breath",
            "winded": "shortness of breath",
            "labored breathing": "shortness of breath",
            
            # Gastrointestinal
            "the runs": "diarrhea",
            "loose stool": "diarrhea",
            "the squirts": "diarrhea",
            
            # Neurological
            "dizzy spells": "dizziness",
            "room spinning": "vertigo",
            "brain fog": "confusion",
            
            # Cardiovascular
            "racing heart": "palpitations",
            "skipped beat": "palpitations",
            
            # General
            "temp": "fever",
            "running hot": "fever",
            "no energy": "fatigue"
        }
    
    def normalize(self, term: str) -> str:
        term = term.strip().lower()
        return self.symptom_map.get(term, term)

normalizer = MedicalNormalizer()

# Symptom database
SYMPTOM_CATEGORIES = {
    "General": [
        "Fever", "Fatigue", "Weight loss", "Weight gain", "Night sweats",
        "Chills", "Weakness", "Malaise", "Loss of appetite", "Dehydration"
    ],
    "Cardiovascular": [
        "Chest pain", "Palpitations", "Shortness of breath", "Swelling in legs",
        "Irregular heartbeat", "Dizziness", "Fainting", "Cold extremities"
    ],
    "Respiratory": [
        "Cough", "Sputum production", "Wheezing", "Chest tightness",
        "Shortness of breath", "Coughing blood", "Nasal congestion",
        "Sore throat", "Hoarseness"
    ],
    "Neurological": [
        "Headache", "Dizziness", "Numbness", "Tingling", "Muscle weakness",
        "Tremors", "Seizures", "Memory loss", "Vision changes", "Loss of balance"
    ],
    "Gastrointestinal": [
        "Abdominal pain", "Nausea", "Vomiting", "Diarrhea", "Constipation",
        "Blood in stool", "Heartburn", "Difficulty swallowing", "Jaundice",
        "Bloating", "Rectal pain"
    ],
    "Musculoskeletal": [
        "Joint pain", "Muscle pain", "Back pain", "Neck pain", "Swollen joints",
        "Limited range of motion", "Morning stiffness", "Muscle cramps"
    ],
    "Dermatological": [
        "Rash", "Itching", "Skin lesions", "Hives", "Hair loss",
        "Nail changes", "Skin discoloration", "Dry skin", "Excessive sweating"
    ],
    "Psychiatric": [
        "Anxiety", "Depression", "Mood swings", "Hallucinations",
        "Confusion", "Insomnia", "Suicidal thoughts", "Panic attacks"
    ],
    "Genitourinary": [
        "Frequent urination", "Painful urination", "Blood in urine",
        "Menstrual irregularities", "Erectile dysfunction", "Vaginal bleeding",
        "Pelvic pain", "Testicular pain"
    ],
    "Other": [
        "Hearing loss", "Tinnitus", "Dental pain", "Swollen lymph nodes",
        "Fainting spells", "Chronic pain", "Sleep disturbances"
    ]
}

EMERGENCY_SYMPTOMS = {
    "en": [
        "Chest pain radiating to arm", "Sudden numbness/weakness", 
        "Difficulty speaking", "Severe head injury", "Suicidal thoughts",
        "Severe allergic reaction", "Uncontrolled bleeding",
        "Loss of consciousness", "Seizure lasting >5 minutes",
        "Severe burns", "Poisoning", "Severe difficulty breathing",
        "Sudden vision loss", "Severe abdominal pain", "High fever with rash"
    ],
    "es": [
        "Dolor de pecho que se irradia al brazo",
        "Entumecimiento/debilidad repentina",
        "Dificultad para hablar",
        "Lesión grave en la cabeza",
        "Pensamientos suicidas",
        "Reacción alérgica grave",
        "Sangrado incontrolable",
        "Pérdida del conocimiento",
        "Convulsiones que duran más de 5 minutos",
        "Quemaduras graves",
        "Envenenamiento",
        "Dificultad respiratoria grave"
    ]
}

LANGUAGE_CONFIG = {
    "en": {"name": "English", "emergency_number": "911"},
    "es": {"name": "Spanish", "emergency_number": "112"}
}

# Database initialization
def init_encrypted_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS assessments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  encrypted_data BLOB,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_encrypted_db()

# LangChain medical analysis
class MedicalAnalysisSystem:
    def __init__(self):
        self.llm = ChatGroq(api_key=GROQ_API_KEY, temperature=0.1, model_name="llama-3.3-70b-specdec")
        self.parser = JsonOutputParser()
        
        self.prompt = ChatPromptTemplate.from_template(
            """Analyze these symptoms: {symptoms}
            Patient description: {patient_input}
            ICD-10 codes: {icd_codes}
            
            Provide JSON with EXACT structure:
            {{
                "differential_diagnosis": [
                    {{
                        "condition": "diagnosis name",
                        "icd10_code": "ICD-10-CM code",
                        "confidence": percentage (0-100)
                    }}
                ],
                "recommended_tests": ["list"],
                "red_flags": ["list"],
                "management_plan": "text",
                "referrals": ["list"],
                "risk_level": "low/moderate/high"
            }}
            Use {language}"""
        )
        
        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, symptoms: List[str], patient_input: str, 
                icd_codes: List[str], language: str) -> Dict:
        try:
            result = self.chain.invoke({
                "symptoms": ", ".join(symptoms),
                "patient_input": patient_input,
                "icd_codes": icd_codes,
                "language": LANGUAGE_CONFIG[language]["name"]
            })
            return self._validate_output(result)
        except Exception as e:
            return self._fallback_response()

    def _validate_output(self, data: Dict) -> Dict:
        validated = {
            "differential_diagnosis": [],
            "recommended_tests": [],
            "red_flags": [],
            "management_plan": "Consult a healthcare provider",
            "referrals": [],
            "risk_level": "unknown"
        }

        # Validate differential diagnosis
        if "differential_diagnosis" in data and isinstance(data["differential_diagnosis"], list):
            for dx in data["differential_diagnosis"]:
                validated["differential_diagnosis"].append({
                    "condition": dx.get("condition", "Unknown condition"),
                    "icd10_code": dx.get("icd10_code", "N/A"),
                    "confidence": min(max(dx.get("confidence", 0), 0), 100)
                })

        # Validate other fields
        for key in ["recommended_tests", "red_flags", "referrals"]:
            if key in data and isinstance(data[key], list):
                validated[key] = data[key][:5]  # Limit to 5 items

        for key in ["management_plan", "risk_level"]:
            if key in data and isinstance(data[key], str):
                validated[key] = data[key][:500]  # Limit length

        return validated

    def _fallback_response(self) -> Dict:
        return {
            "differential_diagnosis": [{
                "condition": "Analysis unavailable",
                "icd10_code": "R69",
                "confidence": 0
            }],
            "recommended_tests": ["Consult a doctor"],
            "red_flags": ["Seek care if symptoms worsen"],
            "management_plan": "Please consult a medical professional",
            "referrals": ["General Physician"],
            "risk_level": "unknown"
        }

# Medical API integration
class ICD10Integrator:
    def __init__(self):
        self.base_url = "http://icd10api.com"
        
    def get_codes(self, symptom: str) -> List[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/?cmd=search&search={symptom}&apikey={ICD10_API_KEY}",
                timeout=10
            )
            return response.json().get("ApproximateMatches", [])[:3]
        except Exception as e:
            st.error(f"ICD-10 API Error: {str(e)}")
            return []

# Data management
class SecureDataHandler:
    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_NAME)
        
    def save_assessment(self, data: Dict):
        encrypted_data = cipher_suite.encrypt(json.dumps(data).encode())
        timestamp = datetime.now(pytz.utc).isoformat()
        self.conn.execute("INSERT INTO assessments (encrypted_data, timestamp) VALUES (?, ?)",
                          (encrypted_data, timestamp))
        self.conn.commit()
        
    def get_history(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT encrypted_data, timestamp FROM assessments ORDER BY timestamp DESC")
        return [
            {"data": json.loads(cipher_suite.decrypt(row[0]).decode()), "timestamp": row[1]}
            for row in cursor.fetchall()
        ]

# Streamlit UI
class HealthcareChatbot:
    def __init__(self):
        self.analyzer = MedicalAnalysisSystem()
        self.integrator = ICD10Integrator()
        self.data_handler = SecureDataHandler()
        
        st.set_page_config(
            page_title="Clinical Symptom Analyzer", 
            page_icon="⚕️", 
            layout="wide"
        )
        
        if "current_language" not in st.session_state:
            st.session_state.current_language = "en"
    
    def _setup_sidebar(self):
        st.sidebar.title("Clinical Tools")
        st.sidebar.selectbox(
            "Language", 
            options=list(LANGUAGE_CONFIG.keys()),
            format_func=lambda x: LANGUAGE_CONFIG[x]["name"],
            key="current_language"
        )
        
        if st.sidebar.button("🩺 Medical History"):
            self._show_medical_history()
            
        if st.sidebar.button("📅 Telemedicine"):
            self._show_telemedicine_scheduler()
    
    def _symptom_selector(self) -> Tuple[List[str], str]:
        with st.form("symptom_form"):
            # Category selection
            col1, col2 = st.columns([1, 2])
            with col1:
                category = st.selectbox(
                    "Symptom Category",
                    options=list(SYMPTOM_CATEGORIES.keys()),
                    key="category"
                )
                
            with col2:
                symptoms = st.multiselect(
                    "Select Symptoms",
                    options=SYMPTOM_CATEGORIES[category],
                    key="symptoms",
                    placeholder="Select or type to search..."
                )
            
            # Additional symptoms
            free_text = st.text_input(
                "Add other symptoms (comma-separated):",
                placeholder="e.g., unusual bruising, metallic taste"
            )
            if free_text:
                symptoms += [s.strip() for s in free_text.split(",")]
            
            # Symptom details
            patient_input = st.text_area(
                "Describe your symptoms (onset, duration, severity):",
                height=150
            )
            
            if st.form_submit_button("Analyze Symptoms"):
                return symptoms, patient_input
        return [], ""
    
    def _handle_emergency(self):
        lang = st.session_state.current_language
        st.error(f"🚨 EMERGENCY ALERT - Call {LANGUAGE_CONFIG[lang]['emergency_number']}")
        st.markdown("""
        **Immediate Actions:**
        1. Stay with the patient
        2. Do not give food/water
        3. Prepare medical information
        4. Follow emergency operator instructions
        """)
        st.markdown("### Emergency Services Activated")
    
    def _display_diagnosis(self, analysis: Dict):
        st.subheader("Clinical Analysis Report")
        
        # Safe access with defaults
        differential_diagnosis = analysis.get("differential_diagnosis", [])
        recommended_tests = analysis.get("recommended_tests", [])
        red_flags = analysis.get("red_flags", [])
        management_plan = analysis.get("management_plan", "No management plan provided")
        referrals = analysis.get("referrals", [])
        risk_level = analysis.get("risk_level", "unknown")

        with st.expander("Differential Diagnosis", expanded=True):
            for dx in differential_diagnosis:
                # Safe access with default values
               condition = dx.get("condition", "Unknown condition")
               icd_code = dx.get("icd10_code", "N/A")
               confidence = dx.get("confidence", 0)
               
               st.markdown(f"**{condition}** (ICD-10: {icd_code})")
               st.caption(f"Confidence: {confidence}%")
               st.progress(min(max(float(confidence)/100, 0), 1))  # Ensure 0-1 range
        
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Recommended Tests")
            for test in analysis["recommended_tests"]:
                st.markdown(f"- {test}")
                
            st.subheader("Red Flags")
            for flag in analysis["red_flags"]:
                st.markdown(f"⚠️ {flag}")
        
        with cols[1]:
            st.subheader("Management Plan")
            st.write(analysis["management_plan"])
            
            st.subheader("Specialist Referrals")
            for referral in analysis["referrals"]:
                st.markdown(f"- {referral}")
    

        # Add risk level display
        risk_color = {
          "high": "red",
          "moderate": "orange",
          "low": "green"
          }.get(risk_level.lower(), "gray")
    
        st.markdown(f"<h3 style='color:{risk_color}'>Risk Level: {risk_level.upper()}</h3>", 
                unsafe_allow_html=True)

    def _show_medical_history(self):
        st.title("Medical History")
        history = self.data_handler.get_history()
        
        if not history:
            st.info("No previous assessments found")
            return
            
        for entry in history[:10]:
            with st.expander(f"Assessment - {entry['timestamp']}"):
                st.json(entry["data"])
    
    def _show_telemedicine_scheduler(self):
        st.title("Virtual Consultation")
        
        with st.form("appointment_form"):
            st.selectbox("Specialist Type", [
                "General Physician", 
                "Cardiologist",
                "Neurologist",
                "Pediatrician",
                "Dermatologist"
            ])
            
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Preferred Date", min_value=datetime.today())
            with col2:
                time = st.time_input("Preferred Time")
            
            if st.form_submit_button("Schedule Appointment"):
                st.success("Appointment Scheduled!")
                st.markdown(f"**Date:** {date.strftime('%Y-%m-%d')}")
                st.markdown(f"**Time:** {time.strftime('%H:%M')}")
                st.markdown("**Meeting Link:** [Secure Video Consultation](https://telemed.example.com)")
    
    def run(self):
        self._setup_sidebar()
        st.title("AI-Powered Clinical Symptom Assessment")
        symptoms, patient_input = self._symptom_selector()
        
        if symptoms:
            # Normalize symptoms using medical terminology
            normalized_symptoms = [normalizer.normalize(s) for s in symptoms]
            
            # Check for emergencies
            if self._is_emergency(normalized_symptoms):
                self._handle_emergency()
                return
                
            with st.spinner("Performing clinical analysis..."):
                # Get ICD-10 codes
                icd_codes = []
                for symptom in normalized_symptoms:
                    codes = self.integrator.get_codes(symptom)
                    icd_codes.extend([c["Code"] for c in codes])
                
                # Perform analysis
                analysis = self.analyzer.analyze(
                    normalized_symptoms, 
                    patient_input,
                    icd_codes,
                    st.session_state.current_language
                )
                
                # Store results
                self.data_handler.save_assessment({
                    "symptoms": normalized_symptoms,
                    "patient_input": patient_input,
                    "analysis": analysis,
                    "timestamp": datetime.now(pytz.utc).isoformat()
                })
                
                # Display results
                self._display_diagnosis(analysis)
    
    def _is_emergency(self, symptoms: List[str]) -> bool:
        lang = st.session_state.current_language
        symptom_text = " ".join(symptoms).lower()
        return any(
            emergency.lower() in symptom_text 
            for emergency in EMERGENCY_SYMPTOMS.get(lang, EMERGENCY_SYMPTOMS["en"])
        )

if __name__ == "__main__":
    chatbot = HealthcareChatbot()
    chatbot.run()
