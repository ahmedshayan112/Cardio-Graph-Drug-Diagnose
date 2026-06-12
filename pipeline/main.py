import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from query_pipeline import PatientFactors, execute_rag_pipeline

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

@app.post("/query")
async def query_rag(factors: PatientFactors):
    try:
        results = execute_rag_pipeline(factors)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
