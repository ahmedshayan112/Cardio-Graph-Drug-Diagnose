import os
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
from query_pipeline import PatientFactors, execute_rag_pipeline

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Missing MONGO_URI in the environment variables.")

# Connect to MongoDB database
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["cardiograph"]

app = FastAPI(
    title="Drug & Disease RAG Diagnosis API",
    description="RAG pipeline querying Pinecone database based on patient factors to retrieve matching drugs & diseases and compute similarity probability.",
    version="1.0.0"
)

# CORS middleware to allow cross-origin requests from frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    patientId: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # 1. Retrieve the patient details from MongoDB
        patient = db["patient_details"].find_one({"patientId": request.patientId})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # 2. Map data to PatientFactors model and run pipeline
        factors = PatientFactors(**patient)
        results = execute_rag_pipeline(factors)
        
        # 3. Store result in drug_diagnosis collection (upsert if exists)
        diagnosis_doc = {
            "patientId": request.patientId,
            "id_1": results["id_1"],
            "id_2": results["id_2"],
            "id_3": results["id_3"]
        }
        db["drug_diagnosis"].replace_one(
            {"patientId": request.patientId},
            diagnosis_doc,
            upsert=True
        )
        
        # Remove MongoDB internal _id before returning (not JSON serializable)
        diagnosis_doc.pop("_id", None)
        
        return diagnosis_doc

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

