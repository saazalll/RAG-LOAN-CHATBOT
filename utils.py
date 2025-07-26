import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FLAN-T5 pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and preprocess CSV into chunks
def load_chunks(csv_path):
    df = pd.read_csv(csv_path).fillna("NA")
    rows = ["\n".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]
    custom_chunks = [
        "Loan_Status is the target variable. It is 'Y' for approved and 'N' for rejected loans.",
        "Married applicants are slightly more likely to get loans, possibly due to combined incomes or coapplicants.",
        "Unmarried applicants with good income and credit history can still get loans approved.",
        "Credit_History is one of the strongest indicators of loan approval in the dataset.",
        "ApplicantIncome and CoapplicantIncome together affect loan eligibility.",
        "Education and Property_Area may also impact loan decisions slightly.",
    ]
    return custom_chunks + rows

# Build FAISS index
def build_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

# Retrieve top relevant chunks
def retrieve(query, index, chunks, k=3):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    return "\n---\n".join([chunks[i] for i in I[0]])

# Generate answer from FLAN-T5
def generate_response(context, query):
    prompt = f"""You are a smart data analyst assistant. Use the dataset context below to answer the user's question clearly and logically.

If the data suggests a trend, explain it briefly. If there's not enough data, say so honestly.

Context:
{context}

Question: {query}

Answer:"""
    return qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"].strip()
