# """
# STEP 5: FastAPI Backend for Legal RAG
# """
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# import os
# import chromadb
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize FastAPI
# app = FastAPI(
#     title="Nepali Legal RAG API",
#     description="Query Nepali legal documents using AI",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:3001"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize RAG components
# api_key = os.getenv('OPENAI_API_KEY')
# openai_client = OpenAI(api_key=api_key)
# chroma_client = chromadb.PersistentClient(path="chroma_db")
# collection = chroma_client.get_collection(name="legal_documents")

# # Request/Response models
# class QueryRequest(BaseModel):
#     query: str
#     n_results: Optional[int] = 5

# class Source(BaseModel):
#     filename: str
#     relevance: float
#     volume: str

# class QueryResponse(BaseModel):
#     query: str
#     answer: str
#     sources: List[Source]

# # Helper functions
# def generate_query_embedding(query: str):
#     response = openai_client.embeddings.create(
#         model="text-embedding-3-large",
#         input=query
#     )
#     return response.data[0].embedding

# def search_documents(query: str, n_results: int):
#     query_embedding = generate_query_embedding(query)
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=n_results
#     )
#     return results

# def generate_answer(query: str, documents: List[str]):
#     context = "\n\n---\n\n".join([
#         f"[Document {i+1}]\n{doc}" 
#         for i, doc in enumerate(documents)
#     ])
    
#     response = openai_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a helpful legal assistant for Nepali law. Answer based ONLY on provided documents. Cite document numbers."},
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"}
#         ],
#         temperature=0.3,
#         max_tokens=1000
#     )
#     return response.choices[0].message.content

# # API Endpoints
# @app.get("/")
# async def root():
#     return {
#         "message": "Nepali Legal RAG API",
#         "version": "1.0.0",
#         "endpoints": {
#             "health": "/health",
#             "query": "/query",
#             "stats": "/stats",
#             "docs": "/docs"
#         }
#     }

# @app.get("/health")
# async def health_check():
#     try:
#         count = collection.count()
#         return {
#             "status": "healthy",
#             "documents_indexed": count
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/stats")
# async def get_stats():
#     try:
#         count = collection.count()
#         return {
#             "total_documents": count,
#             "collection_name": "legal_documents",
#             "model": "text-embedding-3-large"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query", response_model=QueryResponse)
# async def query_documents(request: QueryRequest):
#     try:
#         # Search
#         results = search_documents(request.query, request.n_results)
        
#         documents = results['documents'][0]
#         metadatas = results['metadatas'][0]
#         distances = results['distances'][0]
        
#         # Generate answer
#         answer = generate_answer(request.query, documents)
        
#         # Format sources
#         sources = [
#             Source(
#                 filename=meta.get('filename', 'Unknown'),
#                 relevance=round(1 - dist, 3),
#                 volume=meta.get('volume', 'Unknown')
#             )
#             for meta, dist in zip(metadatas, distances)
#         ]
        
#         return QueryResponse(
#             query=request.query,
#             answer=answer,
#             sources=sources
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


"""
STEP 5: FastAPI Backend for Legal RAG with Citations
"""
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
    version="2.0.0"
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
    Generate answer with proper inline citations
    """
    # Build context with citation markers
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        # Create citation marker with all available info
        citation_parts = [f"Source {i+1}"]
        
        # Add section info if available
        if meta.get('section_type') and meta.get('section_number'):
            section_type = meta['section_type'].replace('_nepali', '').title()
            citation_parts.append(f"{section_type} {meta['section_number']}")
        
        # Add page numbers if available
        if meta.get('page_numbers'):
            pages = meta['page_numbers']
            if isinstance(pages, str):
                try:
                    pages = eval(pages)
                except:
                    pages = []
            
            if pages and len(pages) > 0:
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
4. Use multiple citations if information comes from multiple sources
5. Every factual claim MUST have a citation
6. Be precise and only use information from the provided documents

Examples of proper citations:
✓ "According to [Source 1: Section 5, pp. 12-13], telecommunications require licensing."
✓ "Property rights are defined in [Source 2: Article 12, p. 45]."
✓ "This is supported by [Source 1] and [Source 3: Section 8]."
✓ "The law states [Source 1: धारा 5] that..."

Answer based ONLY on provided documents. Cite everything."""

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
        temperature=0.2,  # Lower for more precise citations
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Nepali Legal RAG API with Citations",
        "version": "2.0.0",
        "features": [
            "Document titles",
            "Section numbers", 
            "Page numbers",
            "Inline citations",
            "Volume information"
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
        return {
            "status": "healthy",
            "documents_indexed": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        count = collection.count()
        
        # Get sample to check citation metadata
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
        # Limit results for better performance
        n_results = min(request.n_results, 10)
        
        # Search
        results = search_documents(request.query, n_results)
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Generate answer with citations
        answer = generate_answer_with_citations(request.query, documents, metadatas)
        
        # Format sources with full citation info
        sources = []
        for meta, dist in zip(metadatas, distances):
            # Handle page_numbers (might be stored as string)
            page_nums = meta.get('page_numbers', [])
            if isinstance(page_nums, str):
                try:
                    page_nums = eval(page_nums)
                except:
                    page_nums = []
            
            # Ensure it's a list
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
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)