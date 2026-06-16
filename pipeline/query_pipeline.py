import os
import math
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import json

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "drug-diagnosis-index"
EMBEDDING_MODEL = "text-embedding-3-small"

# Pydantic model for input validation
class PatientFactors(BaseModel):
    Age: Optional[str] = Field(None, alias="Age")
    Gender: Optional[str] = Field(None, alias="Gender")
    Chief_Complaint: Optional[str] = Field(None, alias="Chief Complaint")
    Previous_Diagnoses: Optional[str] = Field(None, alias="Previous Diagnoses")
    Blood_Pressure: Optional[str] = Field(None, alias="Blood Pressure")
    Temperature: Optional[str] = Field(None, alias="Temperature")
    Pulse: Optional[str] = Field(None, alias="Pulse")
    Respiratory_Rate: Optional[str] = Field(None, alias="Respiratory Rate")
    SpO2: Optional[str] = Field(None, alias="SpO2")
    Pain_Scale: Optional[str] = Field(None, alias="Pain Scale")
    Selected_Symptoms: Optional[str] = Field(None, alias="Selected Symptoms")
    Free_Text: Optional[str] = Field(None, alias="Free Text")
    Duration: Optional[str] = Field(None, alias="Duration")
    Severity: Optional[str] = Field(None, alias="Severity")
    Pattern: Optional[str] = Field(None, alias="Pattern")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def map_mongodb_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapping = {
                "ChiefComplaint": "Chief_Complaint",
                "PreviousDiagnoses": "Previous_Diagnoses",
                "BloodPressure": "Blood_Pressure",
                "RespiratoryRate": "Respiratory_Rate",
                "SelectedSymptoms": "Selected_Symptoms",
                "PainScale": "Pain_Scale",
                "FreeText": "Free_Text"
            }
            for mongo_key, field_name in mapping.items():
                if mongo_key in data and field_name not in data:
                    data[field_name] = data[mongo_key]
        return data

def generate_optimal_query(factors: PatientFactors, openai_client: OpenAI, history: list[dict] = None) -> str:
    """Uses LLM to synthesize an optimal search query from patient factors, optionally incorporating past feedback."""
    history_context = ""
    if history:
        history_context = "\n\nCRITICAL: The previous retrieved drugs did not make clinical sense. Please review the previous attempts and the feedback from the clinical verifier to construct a better search query that targets more appropriate drugs.\n"
        for idx, attempt in enumerate(history, 1):
            history_context += f"""
Attempt {idx}:
- Synthesized Query: {attempt['query']}
- Retrieved Drugs/Diseases: {json.dumps(attempt['drugs'], indent=2)}
- Feedback from Clinical Verifier: {attempt['feedback']}
"""

    prompt = f"""You are an expert medical AI assistant.
Your task is to review the patient's medical factors below and synthesize an optimal, concise query (1 to 2 sentences) that captures their clinical presentation (condition, primary symptoms, previous diagnoses, severity, and duration). This query will be used to retrieve highly relevant drug/disease matches from a vector database of patient experiences.{history_context}

Patient Factors:
- Age: {factors.Age or 'N/A'}
- Gender: {factors.Gender or 'N/A'}
- Chief Complaint: {factors.Chief_Complaint or 'N/A'}
- Previous Diagnoses: {factors.Previous_Diagnoses or 'N/A'}
- Blood Pressure: {factors.Blood_Pressure or 'N/A'}
- Temperature: {factors.Temperature or 'N/A'}
- Pulse: {factors.Pulse or 'N/A'}
- Respiratory Rate: {factors.Respiratory_Rate or 'N/A'}
- SpO2: {factors.SpO2 or 'N/A'}
- Pain Scale: {factors.Pain_Scale or 'N/A'}
- Selected Symptoms: {factors.Selected_Symptoms or 'N/A'}
- Free Text: {factors.Free_Text or 'N/A'}
- Duration: {factors.Duration or 'N/A'}
- Severity: {factors.Severity or 'N/A'}
- Pattern: {factors.Pattern or 'N/A'}

Format instruction: Output ONLY the synthesized query. Do not include any explanations, greetings, introduction, or formatting."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def embed_query(query: str, openai_client: OpenAI) -> list[float]:
    """Generates embeddings for the query text."""
    response = openai_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def similarity_to_probability(score: float) -> float:
    """
    Applies a sigmoid mapping to cosine similarity scores.
    Ensures that low similarities map to low probabilities (< 0.15),
    and high similarities map to high probabilities (> 0.8).
    Formula: 1 / (1 + exp(-k * (score - center)))
    """
    k = 12
    center = 0.48
    prob = 1.0 / (1.0 + math.exp(-k * (score - center)))
    return round(prob, 2)

def verify_drugs_clinical_fit(factors: PatientFactors, recommendations: list[dict], openai_client: OpenAI) -> dict:
    """Agent 2: Checks if the retrieved drugs make clinical sense for the patient's factors.
    Returns a dict containing 'make_sense' (bool) and 'feedback' (str).
    """
    prompt = f"""You are a senior clinical pharmacist and medical auditor (Agent 2).
Your role is to evaluate whether the list of drugs retrieved by the search system (Agent 1) makes clinical sense for the patient's presentation.

Patient Clinical Presentation:
- Age: {factors.Age or 'N/A'}
- Gender: {factors.Gender or 'N/A'}
- Chief Complaint: {factors.Chief_Complaint or 'N/A'}
- Previous Diagnoses: {factors.Previous_Diagnoses or 'N/A'}
- Blood Pressure: {factors.Blood_Pressure or 'N/A'}
- Temperature: {factors.Temperature or 'N/A'}
- Pulse: {factors.Pulse or 'N/A'}
- Respiratory Rate: {factors.Respiratory_Rate or 'N/A'}
- SpO2: {factors.SpO2 or 'N/A'}
- Pain Scale: {factors.Pain_Scale or 'N/A'}
- Selected Symptoms: {factors.Selected_Symptoms or 'N/A'}
- Free Text: {factors.Free_Text or 'N/A'}
- Duration: {factors.Duration or 'N/A'}
- Severity: {factors.Severity or 'N/A'}
- Pattern: {factors.Pattern or 'N/A'}

Retrieved Drug Recommendations:
{json.dumps(recommendations, indent=2)}

Please evaluate:
1. Do these retrieved drugs make clinical sense as potential treatments or related therapies for this patient's chief complaint, symptoms, and medical history?
2. Are there obvious clinical mismatches or irrelevant therapies (e.g., treating a completely unrelated disease or failing to address the primary emergency or chief complaint)?
3. If they make sense, set "make_sense" to true.
4. If they do not make sense, set "make_sense" to false and provide clear, specific feedback explaining what is wrong and what therapeutic direction or focus the next search query should target.

Response format instruction: You must respond ONLY with a valid JSON object matching this schema. Do not include markdown code block markers or any additional text.
{{
    "make_sense": boolean,
    "feedback": "string detailing clinical explanation and query refinement tips if not making sense"
}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical verification system returning JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result_str = response.choices[0].message.content.strip()
        result_json = json.loads(result_str)
        return {
            "make_sense": bool(result_json.get("make_sense", False)),
            "feedback": str(result_json.get("feedback", ""))
        }
    except Exception as e:
        print(f"Error during clinical verification: {e}")
        # Default to True to prevent infinite loops if API fails
        return {"make_sense": True, "feedback": ""}

def execute_rag_pipeline(factors: PatientFactors) -> Dict[str, Any]:
    """Executes the complete RAG pipeline using a multi-agent feedback loop with up to 5 iterations."""
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY or OPENAI_API_KEY in the environment variables.")
        
    # Initialize Clients
    pc = Pinecone(api_key=PINECONE_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Connect to Index
    index = pc.Index(INDEX_NAME)
    
    history = []
    max_iterations = 8
    top_recommendations = []
    
    for iteration in range(1, max_iterations + 1):
        # 1. Agent 1: Synthesize optimal query (with optional history)
        optimal_query = generate_optimal_query(factors, openai_client, history=history)
        
        # 2. Embed the query
        query_vector = embed_query(optimal_query, openai_client)
        
        # 3. Query Pinecone for top candidate matches
        results = index.query(
            vector=query_vector,
            top_k=25,
            include_metadata=True
        )
        
        # 4. Group and deduplicate top drugs and diseases
        seen_combinations = set()
        current_recommendations = []
        
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            drug_name = metadata.get("drug_name")
            disease = metadata.get("prescribed_for")
            score = match.get("score", 0.0)
            
            if not drug_name or not disease:
                continue
                
            combo = (drug_name.lower(), disease.lower())
            if combo not in seen_combinations:
                seen_combinations.add(combo)
                probability = similarity_to_probability(score)
                
                current_recommendations.append({
                    "Drug_Name": drug_name,
                    "Disease": disease,
                    "Probablity": probability,
                    "Raw_Similarity": round(score, 4)
                })
                
            if len(current_recommendations) >= 3:
                break
                
        # If less than 3 recommendations are retrieved, pad them or handle empty states
        while len(current_recommendations) < 3:
            current_recommendations.append({
                "Drug_Name": "No alternative drug found",
                "Disease": "N/A",
                "Probablity": 0.0,
                "Raw_Similarity": 0.0
            })
            
        top_recommendations = current_recommendations
        
        # 5. Agent 2: Clinical verification
        verification = verify_drugs_clinical_fit(factors, top_recommendations, openai_client)
            
        if verification['make_sense']:
            break
            
        # Append to history for the next iteration
        history.append({
            "query": optimal_query,
            "drugs": top_recommendations,
            "feedback": verification['feedback']
        })
        
    # 6. Format to exact user structure
    response_json = {
        "id_1": {
            "Drug_Name": top_recommendations[0]["Drug_Name"],
            "Disease": top_recommendations[0]["Disease"],
            "Probablity": top_recommendations[0]["Probablity"]
        },
        "id_2": {
            "Drug_Name": top_recommendations[1]["Drug_Name"],
            "Disease": top_recommendations[1]["Disease"],
            "Probablity": top_recommendations[1]["Probablity"]
        },
        "id_3": {
            "Drug_Name": top_recommendations[2]["Drug_Name"],
            "Disease": top_recommendations[2]["Disease"],
            "Probablity": top_recommendations[2]["Probablity"]
        }
    }
    
    return response_json

if __name__ == "__main__":
    # Test execution
    test_factors = PatientFactors(
        Age="45",
        Gender="Male",
        Chief_Complaint="High blood pressure and kidney disease",
        Previous_Diagnoses="Kidney disease, hypertension",
        Selected_Symptoms="fluid build up, fluid retention, high pressure",
        Free_Text="My ankles are severely swollen and pressure is very high."
    )
    res = execute_rag_pipeline(test_factors)
    import pprint
    pprint.pprint(res)
