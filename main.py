import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Corrected import
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

print("Starting Vidhi BOT API...")

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("FATAL ERROR: GOOGLE_API_KEY not found in .env file.")
    exit()

app = FastAPI(
    title="विधि BOT API",
    description="API for Nepali Law Chatbot"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI model and vector database...")

DB_PATH = "vectordb"
EMBEDDING_MODEL_NAME = "models/embedding-001"
LLM_NAME = "gemini-pro"

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever()

# Corrected LLM Initialization (argument removed)
llm = GoogleGenerativeAI(model=LLM_NAME, temperature=0.1)

prompt_template = """
You are 'विधि BOT', an expert AI assistant for Nepalese law.
Your task is to answer the user's question in clear, simple language.
Base your answer ONLY on the following legal context provided.
If the answer is not in the context, clearly state that the information is not available in the provided documents.
Cite the source document for the information you use.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("AI components loaded successfully. API is ready.")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    print(f"Received question: {query.question}")
    result = qa_chain({"query": query.question})
    answer = result.get("result")

    source_documents = result.get("source_documents", [])
    sources = []
    if source_documents:
        unique_sources = set(doc.metadata.get('source', 'Unknown') for doc in source_documents)
        sources = [os.path.basename(s) for s in unique_sources]

    print(f"Generated answer: {answer}")
    return {"answer": answer, "sources": sources}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vidhi BOT API!"}