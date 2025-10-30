import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Nepali Legal RAG API with Citations",
    description="Query Nepali legal documents using AI with proper citations",
    version="2.0.4"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
api_key = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="legal_documents")

# Cache for repeated queries (optional)
CACHE = {}

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

class Source(BaseModel):
    filename: str
    relevance: float
    volume: str
    section_type: Optional[str] = None
    section_number: Optional[str] = None
    page_numbers: Optional[List[int]] = None
    citation: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]

# Helper functions
def generate_query_embedding(query: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding

def search_documents(query: str, n_results: int):
    query_embedding = generate_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

def generate_answer_with_citations(query: str, documents: List[str], metadatas: List[dict]):
    """
    Generate deterministic answer with citations.
    Returns special message if query is unrelated.
    """
    if not documents:
        return "माफ गर्नुहोस्, प्रदान गरिएका दस्तावेजहरूमा यो जानकारी उपलब्ध छैन। कृपया कानूनी प्रश्न मात्र सोध्नुहोस्।"

    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        citation_parts = [f"Source {i+1}"]
        if meta.get('section_type') and meta.get('section_number'):
            section_type = meta['section_type'].replace('_nepali', '').title()
            citation_parts.append(f"{section_type} {meta['section_number']}")
        if meta.get('page_numbers'):
            pages = meta['page_numbers']
            if isinstance(pages, str):
                try:
                    pages = eval(pages)
                except:
                    pages = []
            if pages:
                if len(pages) == 1:
                    citation_parts.append(f"p. {pages[0]}")
                elif len(pages) == 2:
                    citation_parts.append(f"pp. {pages[0]}, {pages[1]}")
                else:
                    citation_parts.append(f"pp. {min(pages)}-{max(pages)}")
        marker = f"[{': '.join(citation_parts)}]"
        context_parts.append(f"{marker}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a legal assistant for Nepali law specialized in providing accurate citations.

CRITICAL CITATION RULES:
1. ALWAYS cite sources using brackets: [Source X]
2. Include section numbers when available: [Source X: Section Y]
3. Include page numbers when available: [Source X: Section Y, p. Z]
4. Every factual claim MUST have a citation
5. Use multiple citations if information comes from multiple sources
6. Be precise and only use information from the provided documents

Answer based ONLY on provided documents. If information is not available in documents, respond politely that it's unavailable."""

    user_prompt = f"""Context from legal documents:

{context}

Question: {query}

Provide a comprehensive answer with inline citations for every claim. Use the exact citation format shown in the context."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,           # deterministic
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=1000
    )

    return response.choices[0].message.content

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Nepali Legal RAG API with Citations",
        "version": "2.0.4",
        "features": [
            "Document titles",
            "Section numbers",
            "Page numbers",
            "Inline citations",
            "Volume information",
            "Deterministic answers",
            "Handles unrelated queries",
            "Only includes cited sources"
        ],
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    try:
        count = collection.count()
        return {"status": "healthy", "documents_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        count = collection.count()
        sample = collection.get(limit=1)
        has_citations = False
        if sample and sample['metadatas']:
            has_citations = 'citation' in sample['metadatas'][0]
        return {
            "total_documents": count,
            "collection_name": "legal_documents",
            "model": "text-embedding-3-large",
            "citation_support": has_citations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Check cache first
        if request.query in CACHE:
            return CACHE[request.query]

        n_results = min(request.n_results, 10)
        results = search_documents(request.query, n_results)

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        answer = generate_answer_with_citations(request.query, documents, metadatas)

        # Extract only sources cited in answer (0-based indexing)
        cited_sources = set(
            int(match.group(1)) - 1 for match in re.finditer(r"\[Source (\d+)", answer)
        )

        sources = []
        for i, (meta, dist) in enumerate(zip(metadatas, distances)):
            if i in cited_sources:
                page_nums = meta.get('page_numbers') or []
                if isinstance(page_nums, str):
                    try:
                        page_nums = eval(page_nums)
                    except:
                        page_nums = []
                if not isinstance(page_nums, list):
                    page_nums = []
                sources.append(Source(
                    filename=meta.get('filename', 'Unknown'),
                    relevance=round(1 - dist, 3),
                    volume=meta.get('volume', 'Unknown'),
                    section_type=meta.get('section_type'),
                    section_number=meta.get('section_number'),
                    page_numbers=page_nums,
                    citation=meta.get('citation')
                ))

        response_data = QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources
        )

        # Save to cache
        CACHE[request.query] = response_data
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
