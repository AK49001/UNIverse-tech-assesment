import fastapi
from pydantic import BaseModel
import openai

from utils import generate_embedding, compute_cosine_similarity, upload_to_s3, get_s3_embeddings

# SETTING UP ENV VARIABLES/CONFIGURATIONS
S3_BUCKET = "rag-vector-bucket"
AWS_TYPE = "us-east-1"
openai.api_key = "OPEN_AI_MOCK_KEY"


class ChatRequest(BaseModel):
    message: str


# Building API Endpoints
# Instantiating a fastapi class to build REST API endpoints


app = fastapi.FastAPI()


@app.post("/chat")
async def chat(request: ChatRequest):
    query_embedding = generate_embedding([request.message])
    docs = get_s3_embeddings()

    scored = []
    for doc in docs:
        score = compute_cosine_similarity(query_embedding, doc["embedding"])
        scored.append((score, doc["text"]))

    top_chunks = [t for _, t in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]
    context = "\n".join(top_chunks)

    # Send to OpenAI GPT
    prompt = f"Context:\n{context}\n\nQuestion: {request.message}\nAnswer:"
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": prompt}]
    )
    answer = resp["choices"][0]["message"]["content"]
    return {"answer": answer}


@app.post("/documents/upload")
async def upload_document(file):
    text = (await file.read()).decode("utf-8")
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  

    for idx, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        upload_to_s3(f"{file.filename}_chunk{idx}.json", {
            "text": chunk,
            "embedding": embedding
        })
    return {"status": "uploaded", "chunks": len(chunks)}


@app.get("/documents/search")
async def search_documents(query: str, top_k: int):
    query_embedding = generate_embedding([query])
    docs = get_s3_embeddings()

    all_scores = []
    for doc in docs:
        similarity = compute_cosine_similarity(query_embedding, doc["embedding"])
        all_scores.append((similarity, doc["text"]))

    top_results = sorted(all_scores, key=lambda x: x[0], reverse=True)[:top_k]
    return {"results": [{"score": s, "text": t} for s, t in top_results]}

