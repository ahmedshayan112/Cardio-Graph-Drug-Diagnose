# Cardio Graph Drug Diagnose

A RAG (Retrieval-Augmented Generation) pipeline querying a Pinecone vector database based on patient factors to retrieve matching drugs and diseases, along with similarity probability scores. The backend is exposed via a FastAPI web server.

---

## Architecture & Data Flow

1. **FastAPI Endpoints**: Accepts a POST request at `/query` with patient clinical factor details.
2. **LLM Query Synthesis**: Synthesizes a concise, clinically accurate query (1-2 sentences) using OpenAI `gpt-4o-mini` from the provided patient factors.
3. **Text Embedding**: Generates embeddings using OpenAI's `text-embedding-3-small` model (1536 dimensions).
4. **Vector Database Retrieval**: Queries the Pinecone index `drug-diagnosis-index` to find the top 25 nearest candidate patient experiences.
5. **Probability Calibration**: Calibrates the raw cosine similarity score into a clinically useful probability using a sigmoid function mapping:
   $$\text{Probability} = \frac{1}{1 + e^{-12 \times (\text{score} - 0.48)}}$$
6. **Deduplication & Formatting**: Groups and deduplicates the results by unique (Drug, Disease) combinations and returns the top 3 recommendations.

---

## Technical Stack & Requirements

- **Python Version**: `3.13.12`
- **Primary Dependencies**: FastAPI, Uvicorn, Pinecone, OpenAI, Pydantic, python-dotenv, tqdm, httpx.

---

## Installation & Setup

1. **Clone or Download the Repository**

2. **Set up Virtual Environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # or source .venv/bin/activate  # macOS/Linux
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables Configuration**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY="your-openai-api-key-here"
   PINECONE_API_KEY="your-pinecone-api-key-here"
   ```

---

## Running Data Ingestion (Retriever Setup)

To parse, embed, and upload the local dataset (`Retrever/Drug_Data.json`) to Pinecone:
```bash
python Retrever/ingest_data.py
```
This script will:
- Check for existing Pinecone indexes.
- Create a new serverless index named `drug-diagnosis-index` if it doesn't exist.
- Embed dataset records in batches.
- Upsert metadata and embeddings to Pinecone.

---

## Running the FastAPI Server

To start the API server locally:
```bash
python pipeline/main.py
```
The server will run on `http://127.0.0.1:8000` with auto-reload enabled.

---

## API Documentation & Curl Examples

### Query Endpoint

- **Endpoint**: `/query`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

#### Patient Factors Fields Reference
The model validates inputs matching either python variables (camel_case) or user-friendly aliases:
- `Age`
- `Gender`
- `Chief Complaint`
- `Previous Diagnoses`
- `Blood Pressure`
- `Temperature`
- `Pulse`
- `Respiratory Rate`
- `SpO2`
- `Pain Scale`
- `Selected Symptoms`
- `Free Text`
- `Duration`
- `Severity`
- `Pattern`

#### Example Curl Request

You can send a curl query to the server using the following syntax:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": "45",
       "Gender": "Male",
       "Chief Complaint": "High blood pressure and kidney disease",
       "Previous Diagnoses": "Kidney disease, hypertension",
       "Selected Symptoms": "fluid build up, fluid retention, high pressure",
       "Free Text": "My ankles are severely swollen and pressure is very high."
     }'
```

#### Example JSON Response

```json
{
  "id_1": {
    "Drug_Name": "Furosemide",
    "Probablity": 0.88
  },
  "id_2": {
    "Drug_Name": "Lisinopril",
    "Probablity": 0.76
  },
  "id_3": {
    "Drug_Name": "Metoprolol",
    "Probablity": 0.62
  }
}
```
*(Note: If using `Retrever/query_pipeline.py` standalone script, the return JSON keys also include the matched `Disease` for each recommendation)*.
