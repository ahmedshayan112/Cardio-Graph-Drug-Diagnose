import os
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY or OPENAI_API_KEY in the environment variables.")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_NAME = "drug-diagnosis-index"
DIMENSION = 1536  # text-embedding-3-small dimension
EMBEDDING_MODEL = "text-embedding-3-small"

def get_or_create_index():
    print("Checking Pinecone indexes...")
    # List all indexes
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' does not exist. Creating new serverless index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait until the index is ready
        print("Waiting for index initialization...")
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print(f"Index '{INDEX_NAME}' created and ready.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    return pc.Index(INDEX_NAME)

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "Drug_Data.json")
    print(f"Loading dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_and_upsert(index, data, batch_size=100):
    print(f"Preparing to embed and upsert {len(data)} documents...")
    
    # We will embed batches of text
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        
        # Construct the text representation to embed
        texts_to_embed = []
        vectors_to_upsert = []
        
        for idx, record in enumerate(batch):
            drug_name = record.get("drug_name", "Unknown Drug")
            prescribed_for = record.get("prescribed_for", "Unknown Condition")
            review = record.get("review", "").strip('"').strip()
            
            # Text chunk representation
            text_chunk = f"Disease: {prescribed_for}\nDrug: {drug_name}\nPatient Experience: {review}"
            texts_to_embed.append(text_chunk)
            
            # Keep trace of metadata
            vector_id = f"rec_{i + idx}"
            vectors_to_upsert.append({
                "id": vector_id,
                "metadata": {
                    "drug_name": drug_name,
                    "prescribed_for": prescribed_for,
                    "review": review,
                    "text_chunk": text_chunk
                }
            })
            
        try:
            # Generate embeddings for the batch
            response = openai_client.embeddings.create(
                input=texts_to_embed,
                model=EMBEDDING_MODEL
            )
            
            # Map embeddings to vectors
            for idx, item in enumerate(response.data):
                vectors_to_upsert[idx]["values"] = item.embedding
                
            # Upsert to Pinecone
            index.upsert(vectors=vectors_to_upsert)
            
        except Exception as e:
            print(f"Error during batch {i} to {i + len(batch)}: {e}")
            # Wait a bit and retry
            time.sleep(2)
            try:
                response = openai_client.embeddings.create(
                    input=texts_to_embed,
                    model=EMBEDDING_MODEL
                )
                for idx, item in enumerate(response.data):
                    vectors_to_upsert[idx]["values"] = item.embedding
                index.upsert(vectors=vectors_to_upsert)
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}. Skipping batch.")

    print("Data ingestion completed successfully!")

if __name__ == "__main__":
    index = get_or_create_index()
    data = load_data()
    embed_and_upsert(index, data)
