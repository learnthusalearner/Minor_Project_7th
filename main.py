import os
import pandas as pd
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import gradio as gr
import uvicorn

# ------------------------------
# 1. Load CSV and prepare chunks
# ------------------------------
csv_file = "gita.csv"   # make sure this file is in your repo
df = pd.read_csv(csv_file)
df["text"] = df.fillna("").astype(str).agg(" ".join, axis=1)

def chunk_text(s, size=500):
    return [s[i:i+size] for i in range(0, len(s), size)]

chunks = []
for _, row in df.iterrows():
    chunks.extend(chunk_text(row["text"]))

print(f"✅ Total text chunks: {len(chunks)}")

# ------------------------------
# 2. Embeddings + FAISS
# ------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode(chunks, convert_to_numpy=True)

dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
print("✅ FAISS index ready!")

def retrieve(query, k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

# ------------------------------
# 3. LLM (Phi-3 Mini)
# ------------------------------
model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

chatbot = pipeline("text-generation", model=llm, tokenizer=tokenizer)

# ------------------------------
# 4. Chat function
# ------------------------------
def ask_csv(query, history=[]):
    context = "\n".join(retrieve(query))
    prompt = f"""
You are Lord Krishna, speaking only in the style of the Bhagavad Gita.
Answer briefly (3-5 sentences), with compassion and clarity.
Do NOT invent information — use only the context below.
If unsure, say: "I do not know, but seek within, and clarity shall arise."

Context:
{context}

Arjuna's Question: {query}
Krishna's Guidance:"""

    out = chatbot(prompt, max_new_tokens=200, do_sample=False, temperature=0.2, top_p=0.9)
    answer = out[0]["generated_text"].split("Krishna's Guidance:")[-1].strip()
    return answer

# ------------------------------
# 5. Gradio UI
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📊 CSV Chatbot (Bhagavad Gita AI)")
    gr.ChatInterface(fn=ask_csv)

# ------------------------------
# 6. FastAPI App
# ------------------------------
app = FastAPI()

@app.get("/ask")
def ask_endpoint(q: str = Query(..., description="Your question")):
    answer = ask_csv(q)
    return JSONResponse({"query": q, "answer": answer})

# Mount Gradio at /ui
app = gr.mount_gradio_app(app, demo, path="/ui")

# ------------------------------
# 7. Run (Render/Local)
# ------------------------------
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 1111))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
