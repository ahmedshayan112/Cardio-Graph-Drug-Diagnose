import os
import math
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

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

def generate_optimal_query(factors: PatientFactors, openai_client: OpenAI) -> str:
    """Uses LLM to synthesize an optimal search query from patient factors."""
    prompt = f"""You are an expert medical AI assistant.
Your task is to review the patient's medical factors below and synthesize an optimal, concise query (1 to 2 sentences) that captures their clinical presentation (condition, primary symptoms, previous diagnoses, severity, and duration). This query will be used to retrieve highly relevant drug/disease matches from a vector database of patient experiences.

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

def execute_rag_pipeline(factors: PatientFactors) -> Dict[str, Any]:
    """Executes the complete RAG pipeline and returns the top 3 drugs/diseases."""
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY or OPENAI_API_KEY in the environment variables.")
        
    # Initialize Clients
    pc = Pinecone(api_key=PINECONE_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Connect to Index
    index = pc.Index(INDEX_NAME)
    
    # 1. Synthesize optimal query
    optimal_query = generate_optimal_query(factors, openai_client)
    print(f"Optimal Search Query Synthesized: {optimal_query}")
    
    # 2. Embed the query
    query_vector = embed_query(optimal_query, openai_client)
    
    # 3. Query Pinecone for top candidate matches
    # Fetch top 25 to allow room for deduplication
    results = index.query(
        vector=query_vector,
        top_k=25,
        include_metadata=True
    )
    
    # 4. Group and deduplicate top drugs and diseases
    seen_combinations = set()
    top_recommendations = []
    
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
            
            top_recommendations.append({
                "Drug_Name": drug_name,
                "Disease": disease,
                "Probablity": probability,
                "Raw_Similarity": round(score, 4)
            })
            
        if len(top_recommendations) >= 3:
            break
            
    # If less than 3 recommendations are retrieved, pad them or handle empty states
    while len(top_recommendations) < 3:
        placeholder_idx = len(top_recommendations) + 1
        top_recommendations.append({
            "Drug_Name": "No alternative drug found",
            "Disease": "N/A",
            "Probablity": 0.0,
            "Raw_Similarity": 0.0
        })
        
    # 5. Format to exact user structure
    response_json = {
        "id_1": {
            "Drug_Name": top_recommendations[0]["Drug_Name"],
            "Probablity": top_recommendations[0]["Probablity"]
        },
        "id_2": {
            "Drug_Name": top_recommendations[1]["Drug_Name"],
            "Probablity": top_recommendations[1]["Probablity"]
        },
        "id_3": {
            "Drug_Name": top_recommendations[2]["Drug_Name"],
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
