from fastapi import FastAPI
import chromadb
import ollama 

import os
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
logging.info(f"Using model: {MODEL_NAME}")

# export MODEL_NAME=llama3.1:8b
# uvicorn app:app --host 0.0.0.0 --port 8000


app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")
ollama_client = ollama.Client(host="http://host.docker.internal:11434")

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    # Check if mock mode is enabled
    use_mock = os.getenv("USE_MOCK_LLM", "0") == "1"
    
    if use_mock:
        # Return retrieved context directly (deterministic!)
        return {"answer": context}
    else:
        # Use real LLM (production mode)
        answer = ollama_client.generate(
            model=MODEL_NAME,
            prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:",
        )

        logging.info(f"query asked: {q}")

        return {"answer": answer["response"]}


@app.post("/add")
def add_knowledge(text: str):
    """Add new content to the knowledge base dynamically."""
    try:
        # Generate a unique ID for this document
        import uuid
        doc_id = str(uuid.uuid4())

        # Add the text to chromadb collection
        collection.add(documents=[text], ids=[doc_id])

        logging.info(f"add received new text (id will be generated): {doc_id}")

        return {
            "status": "success",
            "message": "Content added to knowledge base", "id": doc_id}
    except Exception as e:
        return {
            "status": "error",
            "messa": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}
