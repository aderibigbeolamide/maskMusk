import json
import logging
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from flasgger import Swagger

# Load Hugging Face API key from environment variable
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
DOCS_URL = "https://docs.creditchek.africa/"

# Initialize Flask app
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
docs = []
doc_texts = []
doc_embeddings = None
faiss_index = None
embedding_dim = 384  # 'all-MiniLM-L6-v2' outputs 384-dim vectors

# Load documentation passages from a JSON file
def load_docs():
    global docs, doc_texts
    docs = []
    doc_texts = []

    # Try different paths where docs.json might be located
    possible_paths = [
        "docs.json",  # Default location in the same directory
        os.path.join(os.getcwd(), "docs.json"),  # Absolute path in working directory
        "/opt/render/project/src/docs.json",  # Render deployment path
    ]

    for file_path in possible_paths:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                doc_texts = [doc["text"] for doc in docs]
                logger.info(f"✅ Loaded {len(doc_texts)} documentation passages from {file_path}.")
                return  # Exit after successful load
            except Exception as e:
                logger.error(f"❌ Error loading docs from {file_path}: {e}")

    logger.warning("⚠️ No documentation passages found. Make sure docs.json is available.")

# Build FAISS index using SentenceTransformer embeddings
def build_faiss_index():
    global faiss_index, doc_embeddings, doc_texts, embedding_dim
    if not doc_texts:
        logger.warning("No documentation passages available. Skipping FAISS index build.")
        return
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Computing embeddings for documentation passages...")
    doc_embeddings = model.encode(doc_texts, convert_to_numpy=True)
    
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(doc_embeddings)
    logger.info("FAISS index built successfully.")

# Perform vector search in local docs.json
def vector_search(query, k=5):
    if faiss_index is None:
        logger.warning("FAISS index not initialized. Returning empty response.")
        return None
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)
    
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(doc_texts) and distance < 1.0:  # Filter out irrelevant results
            results.append(doc_texts[idx])

    return "\n".join(results) if results else None


# Fetch additional data from external documentation
def fetch_external_docs(query):
    try:
        response = requests.get(f"{DOCS_URL}/search?q={query}")
        response.raise_for_status()
        return response.text[:1000]  # Limit the response length
    except requests.RequestException as e:
        logger.error(f"Error fetching external docs: {e}")
        return None

# Query LLM using Hugging Face API
def query_llm(prompt):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.5}}

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and result:
            generated_text = result[0].get("generated_text", "")
            
            # Extract only the part that directly answers the query
            return generated_text.split("Implementation Details")[0].strip()

        return "Unexpected API response"
    except requests.RequestException as e:
        return f"Error: Unable to get response from API - {e}"



@app.route('/')
def home():
    return jsonify({"message": "API is running. Use /ask to query."})

@app.route('/ask', methods=['POST'])
def ask():
    """
    API for querying documentation with AI assistance.
    ---
    tags:
      - Query API
    parameters:
      - name: query
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              example: "How do I integrate CreditChek API?"
    responses:
      200:
        description: AI-generated response with relevant documentation and sample code
        schema:
          type: object
          properties:
            answer:
              type: string
              example: "To integrate CreditChek API, use the following..."
      400:
        description: Bad request when no query is provided.
      404:
        description: No relevant information found.
    """
    data = request.get_json()
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Step 1: Search in local docs.json
    context = vector_search(query)
    
    # Step 2: If not found locally, check external docs
    if not context:
        logger.info("No relevant data found in local JSON. Searching external docs...")
        context = fetch_external_docs(query)

    # Step 3: If still nothing, return 404
    if not context:
        return jsonify({"error": "No relevant information found."}), 404

    # Step 4: Prioritize clear definitions before implementation details
    prompt = (
    f"### Question: {query}\n"
    f"### Relevant Context:\n{context}\n\n"
    "### Answer concisely and specifically, avoiding unnecessary details."
    )
    answer = query_llm(prompt)


    return jsonify({"answer": answer})

if __name__ == '__main__':
    load_docs("docs.json")
    
    if doc_texts:
        build_faiss_index()
    else:
        logger.warning("No documentation passages loaded. Vector search will not work correctly.")
    
    app.run(debug=True)
