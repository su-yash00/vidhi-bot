# # """
# # STEP 5: FastAPI Backend for Legal RAG with Citations
# # """
# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from typing import List, Optional
# # import os
# # import chromadb
# # from openai import OpenAI
# # from dotenv import load_dotenv

# # load_dotenv()

# # # Initialize FastAPI
# # app = FastAPI(
# #     title="Nepali Legal RAG API with Citations",
# #     description="Query Nepali legal documents using AI with proper citations",
# #     version="2.0.0"
# # )

# # # CORS middleware
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["http://localhost:3000", "http://localhost:3001"],  
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Initialize RAG components
# # api_key = os.getenv('OPENAI_API_KEY')
# # openai_client = OpenAI(api_key=api_key)
# # chroma_client = chromadb.PersistentClient(path="chroma_db")
# # collection = chroma_client.get_collection(name="legal_documents")

# # # Request/Response models
# # class QueryRequest(BaseModel):
# #     query: str
# #     n_results: Optional[int] = 5

# # class Source(BaseModel):
# #     filename: str
# #     relevance: float
# #     volume: str
# #     section_type: Optional[str] = None
# #     section_number: Optional[str] = None
# #     page_numbers: Optional[List[int]] = None
# #     citation: Optional[str] = None

# # class QueryResponse(BaseModel):
# #     query: str
# #     answer: str
# #     sources: List[Source]

# # # Helper functions
# # def generate_query_embedding(query: str):
# #     response = openai_client.embeddings.create(
# #         model="text-embedding-3-large",
# #         input=query
# #     )
# #     return response.data[0].embedding

# # def search_documents(query: str, n_results: int):
# #     query_embedding = generate_query_embedding(query)
# #     results = collection.query(
# #         query_embeddings=[query_embedding],
# #         n_results=n_results
# #     )
# #     return results

# # def generate_answer_with_citations(query: str, documents: List[str], metadatas: List[dict]):
# #     """
# #     Generate answer with proper inline citations
# #     """
# #     # Build context with citation markers
# #     context_parts = []
# #     for i, (doc, meta) in enumerate(zip(documents, metadatas)):
# #         # Create citation marker with all available info
# #         citation_parts = [f"Source {i+1}"]
        
# #         # Add section info if available
# #         if meta.get('section_type') and meta.get('section_number'):
# #             section_type = meta['section_type'].replace('_nepali', '').title()
# #             citation_parts.append(f"{section_type} {meta['section_number']}")
        
# #         # Add page numbers if available
# #         if meta.get('page_numbers'):
# #             pages = meta['page_numbers']
# #             if isinstance(pages, str):
# #                 try:
# #                     pages = eval(pages)
# #                 except:
# #                     pages = []
            
# #             if pages and len(pages) > 0:
# #                 if len(pages) == 1:
# #                     citation_parts.append(f"p. {pages[0]}")
# #                 elif len(pages) == 2:
# #                     citation_parts.append(f"pp. {pages[0]}, {pages[1]}")
# #                 else:
# #                     citation_parts.append(f"pp. {min(pages)}-{max(pages)}")
        
# #         marker = f"[{': '.join(citation_parts)}]"
# #         context_parts.append(f"{marker}\n{doc}")
    
# #     context = "\n\n---\n\n".join(context_parts)
    
# #     system_prompt = """You are a legal assistant for Nepali law specialized in providing accurate citations.

# # CRITICAL CITATION RULES:
# # 1. ALWAYS cite sources using brackets: [Source X]
# # 2. Include section numbers when available: [Source X: Section Y]
# # 3. Include page numbers when available: [Source X: Section Y, p. Z]
# # 4. Use multiple citations if information comes from multiple sources
# # 5. Every factual claim MUST have a citation
# # 6. Be precise and only use information from the provided documents

# # Examples of proper citations:
# # ✓ "According to [Source 1: Section 5, pp. 12-13], telecommunications require licensing."
# # ✓ "Property rights are defined in [Source 2: Article 12, p. 45]."
# # ✓ "This is supported by [Source 1] and [Source 3: Section 8]."
# # ✓ "The law states [Source 1: धारा 5] that..."

# # Answer based ONLY on provided documents. Cite everything."""

# #     user_prompt = f"""Context from legal documents:

# # {context}

# # Question: {query}

# # Provide a comprehensive answer with inline citations for every claim. Use the exact citation format shown in the context."""

# #     response = openai_client.chat.completions.create(
# #         model="gpt-4o-mini",
# #         messages=[
# #             {"role": "system", "content": system_prompt},
# #             {"role": "user", "content": user_prompt}
# #         ],
# #         temperature=0.2,  # Lower for more precise citations
# #         max_tokens=1000
# #     )
    
# #     return response.choices[0].message.content

# # # API Endpoints
# # @app.get("/")
# # async def root():
# #     return {
# #         "message": "Nepali Legal RAG API with Citations",
# #         "version": "2.0.0",
# #         "features": [
# #             "Document titles",
# #             "Section numbers", 
# #             "Page numbers",
# #             "Inline citations",
# #             "Volume information"
# #         ],
# #         "endpoints": {
# #             "health": "/health",
# #             "query": "/query",
# #             "stats": "/stats",
# #             "docs": "/docs"
# #         }
# #     }

# # @app.get("/health")
# # async def health_check():
# #     try:
# #         count = collection.count()
# #         return {
# #             "status": "healthy",
# #             "documents_indexed": count
# #         }
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.get("/stats")
# # async def get_stats():
# #     try:
# #         count = collection.count()
        
# #         # Get sample to check citation metadata
# #         sample = collection.get(limit=1)
# #         has_citations = False
# #         if sample and sample['metadatas']:
# #             has_citations = 'citation' in sample['metadatas'][0]
        
# #         return {
# #             "total_documents": count,
# #             "collection_name": "legal_documents",
# #             "model": "text-embedding-3-large",
# #             "citation_support": has_citations
# #         }
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/query", response_model=QueryResponse)
# # async def query_documents(request: QueryRequest):
# #     try:
# #         # Limit results for better performance
# #         n_results = min(request.n_results, 10)
        
# #         # Search
# #         results = search_documents(request.query, n_results)
        
# #         documents = results['documents'][0]
# #         metadatas = results['metadatas'][0]
# #         distances = results['distances'][0]
        
# #         # Generate answer with citations
# #         answer = generate_answer_with_citations(request.query, documents, metadatas)
        
# #         # Format sources with full citation info
# #         sources = []
# #         for meta, dist in zip(metadatas, distances):
# #             # Handle page_numbers (might be stored as string)
# #             page_nums = meta.get('page_numbers', [])
# #             if isinstance(page_nums, str):
# #                 try:
# #                     page_nums = eval(page_nums)
# #                 except:
# #                     page_nums = []
            
# #             # Ensure it's a list
# #             if not isinstance(page_nums, list):
# #                 page_nums = []
            
# #             sources.append(Source(
# #                 filename=meta.get('filename', 'Unknown'),
# #                 relevance=round(1 - dist, 3),
# #                 volume=meta.get('volume', 'Unknown'),
# #                 section_type=meta.get('section_type'),
# #                 section_number=meta.get('section_number'),
# #                 page_numbers=page_nums,
# #                 citation=meta.get('citation')
# #             ))
        
# #         return QueryResponse(
# #             query=request.query,
# #             answer=answer,
# #             sources=sources
# #         )
        
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


# """
# STEP 5: Enhanced FastAPI Backend with Relevance Filtering
# """
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional, Dict
# import os
# import json
# import chromadb
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI(
#     title="Nepali Legal RAG API - Enhanced",
#     description="Query Nepali legal documents with query reformulation and comprehensive citations",
#     version="3.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize components
# api_key = os.getenv('OPENAI_API_KEY')
# openai_client = OpenAI(api_key=api_key)
# chroma_client = chromadb.PersistentClient(path="chroma_db")
# collection = chroma_client.get_collection(name="legal_documents")

# # Models
# class QueryRequest(BaseModel):
#     query: str
#     n_results_per_query: Optional[int] = 3
#     use_reformulation: Optional[bool] = True
#     include_summary: Optional[bool] = True

# class Source(BaseModel):
#     index: int
#     filename: str
#     document_title: Optional[str] = None
#     relevance: float
#     volume: str
#     section_type: Optional[str] = None
#     section_number: Optional[str] = None
#     page_numbers: Optional[List[int]] = None
#     citation: Optional[str] = None

# class QueryResponse(BaseModel):
#     query: str
#     query_variations: Optional[List[str]] = None
#     answer: str
#     summary: Optional[str] = None
#     sources: List[Source]
#     total_sources: int
#     out_of_context: Optional[bool] = False

# # Helper Functions
# def check_query_relevance(query: str) -> tuple[bool, str]:
#     """
#     Check if the query is relevant to Nepali legal documents
#     Returns: (is_relevant, reason)
#     """
#     prompt = f"""You are a query relevance checker for a Nepali legal document database.

# The database contains:
# - Nepali laws, acts, and regulations
# - Legal provisions and statutes
# - Government policies and legal frameworks
# - Constitutional articles
# - Legal procedures and requirements

# Query: "{query}"

# Determine if this query is asking about:
# 1. Nepali legal matters, laws, or regulations
# 2. Legal procedures or requirements in Nepal
# 3. Constitutional provisions
# 4. Government policies or legal frameworks
# 5. Legal definitions or interpretations

# Respond with ONLY:
# - "RELEVANT" if the query is about legal matters
# - "IRRELEVANT" if the query is about personal information, general knowledge, greetings, or non-legal topics

# Format: RELEVANT or IRRELEVANT"""

#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a query relevance classifier."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.1,
#             max_tokens=50
#         )
#         result = response.choices[0].message.content.strip().upper()
        
#         is_relevant = "RELEVANT" in result
#         reason = "Query is about legal matters" if is_relevant else "Query is not related to legal documents"
        
#         return is_relevant, reason
        
#     except Exception as e:
#         print(f"Error checking relevance: {e}")
#         # Default to allowing the query if check fails
#         return True, "Relevance check failed, allowing query"

# def reformulate_query(original_query: str) -> List[str]:
#     """Generate query variations"""
#     prompt = f"""Given this legal query, generate 3 different variations that capture the same intent.
# Consider technical legal terminology, Nepali and English variants, and different phrasings.

# Original query: {original_query}

# Provide ONLY the 3 variations, one per line."""

#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a legal query reformulation expert."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=200
#         )
#         variations = response.choices[0].message.content.strip().split('\n')
#         variations = [v.strip() for v in variations if v.strip()]
#         return [original_query] + variations[:3]
#     except:
#         return [original_query]

# def generate_query_embedding(query: str):
#     """Generate embedding"""
#     response = openai_client.embeddings.create(
#         model="text-embedding-3-large",
#         input=query
#     )
#     return response.data[0].embedding

#  def search_with_reformulation(query: str, n_results_per_query: int, use_reformulation: bool):
#     """Search using query variations with relevance threshold"""
#     if use_reformulation:
#         query_variations = reformulate_query(query)
#     else:
#         query_variations = [query]
    
#     all_results = {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}
#     seen_ids = set()
    
#     # RELEVANCE THRESHOLD - documents must be at least this similar (lower distance = more similar)
#     RELEVANCE_THRESHOLD = 0.7  # Adjust this value (0.5-0.9 range)
    
#     for i, q_var in enumerate(query_variations):
#         query_embedding = generate_query_embedding(q_var)
#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=n_results_per_query
#         )
        
#         for j in range(len(results['ids'][0])):
#             doc_id = results['ids'][0][j]
#             distance = results['distances'][0][j]
            
#             # Skip documents that are too dissimilar (distance too high)
#             if distance > RELEVANCE_THRESHOLD:
#                 continue
                
#             if doc_id not in seen_ids:
#                 seen_ids.add(doc_id)
#                 all_results['documents'].append(results['documents'][0][j])
#                 all_results['metadatas'].append(results['metadatas'][0][j])
#                 adjusted_distance = distance * (1 + i * 0.1)
#                 all_results['distances'].append(adjusted_distance)
#                 all_results['ids'].append(doc_id)
    
#     # If no relevant documents found, return empty results
#     if not all_results['documents']:
#         return {
#             'documents': [[]],
#             'metadatas': [[]],
#             'distances': [[]],
#             'ids': [[]]
#         }, query_variations
    
#     sorted_indices = sorted(range(len(all_results['distances'])), 
#                            key=lambda i: all_results['distances'][i])
#     max_results = min(10, len(sorted_indices))
#     sorted_indices = sorted_indices[:max_results]
    
#     return {
#         'documents': [[all_results['documents'][i] for i in sorted_indices]],
#         'metadatas': [[all_results['metadatas'][i] for i in sorted_indices]],
#         'distances': [[all_results['distances'][i] for i in sorted_indices]],
#         'ids': [[all_results['ids'][i] for i in sorted_indices]]
#     }, query_variations


# def summarize_chunks(documents: List[str]) -> str:
#     """Summarize retrieved content"""
#     combined_text = "\n\n".join(documents[:5])
#     prompt = f"""Summarize the key legal points from these excerpts in 3-4 bullet points:

# {combined_text}

# Provide a concise summary highlighting the main legal provisions."""

#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a legal document summarizer."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=300
#         )
#         return response.choices[0].message.content
#     except:
#         return None

# def extract_relevant_quotes(documents: List[str]) -> List[str]:
#     """Extract relevant quotes"""
#     quotes = []
#     for doc in documents[:5]:
#         sentences = doc.split('.')
#         good_sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 50]
#         if good_sentences:
#             quotes.append(good_sentences[0])
#     return quotes

# def generate_answer_with_citations(query: str, documents: List[str], metadatas: List[dict]):
#     """Generate comprehensive answer"""
#     context_parts = []
#     for i, (doc, meta) in enumerate(zip(documents, metadatas)):
#         citation_parts = [f"[{i+1}]"]
        
#         if meta.get('document_title'):
#             citation_parts.append(meta['document_title'][:50])
        
#         if meta.get('section_type') and meta.get('section_number'):
#             section = meta['section_type'].replace('_nepali', '').title()
#             citation_parts.append(f"{section} {meta['section_number']}")
        
#         if meta.get('page_numbers'):
#             pages = meta['page_numbers']
#             if isinstance(pages, str):
#                 try:
#                     pages = json.loads(pages)
#                 except:
#                     pages = []
#             if pages and len(pages) > 0:
#                 if len(pages) == 1:
#                     citation_parts.append(f"p.{pages[0]}")
#                 else:
#                     citation_parts.append(f"pp.{min(pages)}-{max(pages)}")
        
#         marker = " ".join(citation_parts)
#         context_parts.append(f"{marker}\n{doc}\n")
    
#     context = "\n---\n".join(context_parts)
#     quotes = extract_relevant_quotes(documents)
#     quotes_section = "\n".join([f"• \"{q}\"" for q in quotes[:3]])
    
#     system_prompt = """You are an expert legal assistant for Nepali law.

# CITATION RULES:
# 1. ALWAYS cite sources using [Number]
# 2. Place citations after claims: "The law states X [1]"
# 3. Use multiple citations: "X [1] and Y [2]"
# 4. Every factual claim MUST have a citation
# 5. Start with a direct answer
# 6. Provide supporting details
# 7. End with key points summary

# Answer based ONLY on provided documents."""

#     user_prompt = f"""Context:

# {context}

# Relevant quotes:
# {quotes_section}

# Question: {query}

# Provide:
# 1. Direct answer with citations
# 2. Supporting details with citations
# 3. Relevant provisions
# 4. Key points summary"""

#     response = openai_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=1500
#     )
    
#     return response.choices[0].message.content

# # API Endpoints
# @app.get("/")
# async def root():
#     return {
#         "message": "Nepali Legal RAG API - Enhanced Version",
#         "version": "3.0.0",
#         "features": [
#             "Query relevance filtering",
#             "Query reformulation",
#             "Chunk summarization",
#             "Enhanced citations (titles, sections, pages)",
#             "Relevant quote extraction",
#             "Improved retrieval quality"
#         ]
#     }

# @app.get("/health")
# async def health_check():
#     try:
#         count = collection.count()
#         return {"status": "healthy", "documents_indexed": count}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/stats")
# async def get_stats():
#     try:
#         count = collection.count()
#         sample = collection.get(limit=1)
#         metadata_fields = []
#         if sample and sample['metadatas']:
#             metadata_fields = list(sample['metadatas'][0].keys())
        
#         return {
#             "total_documents": count,
#             "collection_name": "legal_documents",
#             "model": "text-embedding-3-large",
#             "version": "3.0.0",
#             "available_metadata": metadata_fields
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query", response_model=QueryResponse)
# async def query_documents(request: QueryRequest):
#     try:
#         # STEP 1: Check query relevance
#         is_relevant, reason = check_query_relevance(request.query)
        
#         if not is_relevant:
#             return QueryResponse(
#                 query=request.query,
#                 query_variations=None,
#                 answer="I can only answer questions about Nepali legal documents, laws, acts, regulations, and legal procedures. Your question appears to be outside this scope.",
#                 summary=None,
#                 sources=[],
#                 total_sources=0,
#                 out_of_context=True
#             )
        
#         # STEP 2: Search with reformulation and relevance threshold
#         n_results = min(request.n_results_per_query, 5)
#         results, query_variations = search_with_reformulation(
#             request.query, 
#             n_results, 
#             request.use_reformulation
#         )
        
#         documents = results['documents'][0]
#         metadatas = results['metadatas'][0]
#         distances = results['distances'][0]
        
#         # STEP 3: Check if any relevant documents found
#         if not documents or len(documents) == 0:
#             return QueryResponse(
#                 query=request.query,
#                 query_variations=query_variations if request.use_reformulation else None,
#                 answer="I couldn't find any relevant information in the Nepali legal documents database for your query. Please try rephrasing your question or ask about specific laws, acts, or legal provisions.",
#                 summary=None,
#                 sources=[],
#                 total_sources=0,
#                 out_of_context=True
#             )
        
#         # STEP 4: Generate summary if requested
#         summary = None
#         if request.include_summary:
#             summary = summarize_chunks(documents)
        
#         # STEP 5: Generate answer
#         answer = generate_answer_with_citations(request.query, documents, metadatas)
        
#         # STEP 6: Format sources
#         sources = []
#         for i, (meta, dist) in enumerate(zip(metadatas, distances), 1):
#             page_nums = meta.get('page_numbers', [])
#             if isinstance(page_nums, str):
#                 try:
#                     page_nums = json.loads(page_nums)
#                 except:
#                     page_nums = []
#             if not isinstance(page_nums, list):
#                 page_nums = []
            
#             sources.append(Source(
#                 index=i,
#                 filename=meta.get('filename', 'Unknown'),
#                 document_title=meta.get('document_title'),
#                 relevance=round(1 - dist, 3),
#                 volume=meta.get('volume', 'Unknown'),
#                 section_type=meta.get('section_type'),
#                 section_number=meta.get('section_number'),
#                 page_numbers=page_nums,
#                 citation=meta.get('citation')
#             ))
        
#         return QueryResponse(
#             query=request.query,
#             query_variations=query_variations if request.use_reformulation else None,
#             answer=answer,
#             summary=summary,
#             sources=sources,
#             total_sources=len(sources),
#             out_of_context=False
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



"""
STEP 5: Enhanced FastAPI Backend with Relevance Filtering
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import re # Import re for cleaning query variations

load_dotenv()

app = FastAPI(
    title="Nepali Legal RAG API - Enhanced",
    description="Query Nepali legal documents with query reformulation and comprehensive citations",
    version="3.0.4"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
api_key = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="legal_documents")

# Models
class QueryRequest(BaseModel):
    query: str
    n_results_per_query: Optional[int] = 7
    use_reformulation: Optional[bool] = True
    include_summary: Optional[bool] = True

class Source(BaseModel):
    index: int
    filename: str
    document_title: Optional[str] = None
    relevance: float
    volume: str
    section_type: Optional[str] = None
    section_number: Optional[str] = None
    page_numbers: Optional[List[int]] = None
    citation: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    query_variations: Optional[List[str]] = None
    answer: str
    summary: Optional[str] = None
    sources: List[Source]
    total_sources: int
    out_of_context: Optional[bool] = False

# Helper Functions
def check_query_relevance(query: str) -> tuple[bool, str]:
    """
    Check if the query is relevant to the *content* of Nepali legal documents.
    Returns: (is_relevant, reason)
    """
    prompt = f"""You are a query classifier for a database of **Nepali legal documents (Acts, Laws, Regulations)**.
Your job is to determine if a query is asking for information **likely to be found inside a specific legal text file**.

- **RELEVANT** queries ask about the *content* of laws, acts, or regulations:
    - "What is the penalty for theft in the Muluki Ain?"
    - "What does the Telecommunications Act say about licensing?"
    - "Provisions for citizenship in Nepal's constitution"
    - "What is the legal definition of 'cybercrime'?"
    - "What acts are considered cybercrimes?"

- **IRRELEVANT** queries ask for general knowledge, news, or specific facts *not* found in a legal code:
    - "Who is the prime minister of Nepal?"
    - "What is the capital of Nepal?"
    - "When is the next election?"
    - "What is the weather today?"

Query: "{query}"

Based on these rules, is the query RELEVANT or IRRELEVANT?
Respond with ONLY: "RELEVANT" or "IRRELEVANT".
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict query relevance classifier for a legal database."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=20
        )
        result = response.choices[0].message.content.strip().upper()
        
        is_relevant = "RELEVANT" in result
        reason = "Query is relevant to legal documents." if is_relevant else "Query is general knowledge and not found in legal texts."
        
        print(f"DEBUG: Relevance check for '{query}': {result}")
        return is_relevant, reason
        
    except Exception as e:
        print(f"Error checking relevance: {e}")
        # Default to allowing the query if check fails, but log it
        return True, "Relevance check failed, defaulting to relevant"

def reformulate_query(original_query: str) -> List[str]:
    """Generate query variations, including specific Nepali terms."""
    prompt = f"""Given this legal query for a bilingual (English/Nepali) database, generate 3 optimized search queries.
Include variations with specific Nepali legal terms to improve retrieval.

Examples:
- Query: "fundamental rights" -> "मौलिक हक" (Maulik Hak)
- Query: "broadcasting" -> "प्रसारण" (Prasaran) or "राष्ट्रिय प्रसारण ऐन" (National Broadcasting Act)
- Query: "telecommunications" -> "दूरसञ्चार" (Dursanchar) or "दूरसञ्चार ऐन" (Telecommunications Act)
- Query: "cybercrime" -> "विद्युतीय कारोबार ऐन" (Electronic Transaction Act) or "साइबर अपराध"

Original query: {original_query}

Provide ONLY the 3 variations, one per line. Do not number them.
Focus on keywords and bilingual terms.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal query reformulation expert for a bilingual English/Nepali database."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        variations = response.choices[0].message.content.strip().split('\n')
        # Clean up numbering like "1. "
        variations = [re.sub(r'^\d+\.\s*', '', v.strip()) for v in variations if v.strip()]
        
        print(f"DEBUG: Reformulated queries: {variations}")
        return [original_query] + variations[:3]
        
    except Exception as e:
        print(f"Error reformulating query: {e}")
        return [original_query]

def generate_query_embedding(query: str):
    """Generate embedding"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding

def search_with_reformulation(query: str, n_results_per_query: int, use_reformulation: bool):
    """
    Search using query variations.
    We retrieve the top documents and let the LLM's strict prompt
    decide if they are relevant.
    """
    if use_reformulation:
        query_variations = reformulate_query(query)
    else:
        query_variations = [query]
    
    all_results = {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}
    seen_ids = set()
    
    print(f"\n🔍 DEBUG: Searching with {len(query_variations)} variations. (No distance threshold)")
    
    for i, q_var in enumerate(query_variations):
        print(f"  Query {i+1}: {q_var[:60]}...")
        query_embedding = generate_query_embedding(q_var)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_per_query
        )
        
        print(f"  → Found {len(results['ids'][0])} results from ChromaDB")
        if results['distances'] and results['distances'][0]:
            print(f"  → Distance range: {min(results['distances'][0]):.3f} - {max(results['distances'][0]):.3f}")
        
        for j in range(len(results['ids'][0])):
            doc_id = results['ids'][0][j]
            distance = results['distances'][0][j]
            
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results['documents'].append(results['documents'][0][j])
                all_results['metadatas'].append(results['metadatas'][0][j])
                # Give a slight penalty to variations to prioritize the original query's results
                adjusted_distance = distance * (1 + i * 0.1) 
                all_results['distances'].append(adjusted_distance)
                all_results['ids'].append(doc_id)
                print(f"    ✓ Added doc {doc_id[:20]}... (distance: {distance:.3f})")
    
    print(f"\n  Total unique documents collected: {len(all_results['documents'])}")
    
    if not all_results['documents']:
        print("  ❌ NO DOCUMENTS FOUND!")
        return {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]],
            'ids': [[]]
        }, query_variations
    
    # Sort by the adjusted distance
    sorted_indices = sorted(range(len(all_results['distances'])), 
                           key=lambda i: all_results['distances'][i])
    
    # Keep up to 10 top documents for the LLM
    max_results = min(10, len(sorted_indices))
    sorted_indices = sorted_indices[:max_results]
    
    print(f"  ✓ Returning top {max_results} documents to LLM")
    
    return {
        'documents': [[all_results['documents'][i] for i in sorted_indices]],
        'metadatas': [[all_results['metadatas'][i] for i in sorted_indices]],
        'distances': [[all_results['distances'][i] for i in sorted_indices]],
        'ids': [[all_results['ids'][i] for i in sorted_indices]]
    }, query_variations

def summarize_chunks(documents: List[str]) -> str:
    """Summarize retrieved content"""
    combined_text = "\n\n".join(documents[:5])
    prompt = f"""Summarize the key legal points from these excerpts in 3-4 bullet points:

{combined_text}

Provide a concise summary highlighting the main legal provisions."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal document summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except:
        return None

def extract_relevant_quotes(documents: List[str]) -> List[str]:
    """Extract relevant quotes"""
    quotes = []
    for doc in documents[:5]:
        sentences = doc.split('.')
        good_sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 50]
        if good_sentences:
            quotes.append(good_sentences[0])
    return quotes

def generate_answer_with_citations(query: str, documents: List[str], metadatas: List[dict]):
    """Generate comprehensive answer with strict context-adherence and synthesis."""
    
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        # Build the citation marker
        citation_parts = [f"[{i+1}]"]
        if meta.get('document_title'):
            citation_parts.append(meta['document_title'][:50])
        if meta.get('section_type') and meta.get('section_number'):
            section = meta['section_type'].replace('_nepali', '').title()
            citation_parts.append(f"{section} {meta['section_number']}")
        if meta.get('page_numbers'):
            pages = meta.get('page_numbers', [])
            if isinstance(pages, str):
                try: pages = json.loads(pages)
                except: pages = []
            if pages and len(pages) > 0:
                if len(pages) == 1: citation_parts.append(f"p.{pages[0]}")
                else: citation_parts.append(f"pp.{min(pages)}-{max(pages)}")
        
        marker = " ".join(citation_parts)
        context_parts.append(f"--- START OF SOURCE {marker} ---\n{doc}\n--- END OF SOURCE {marker} ---\n")
    
    context = "\n".join(context_parts)
    
    # --- THIS IS THE CRUCIAL CHANGE ---
    system_prompt = """You are a highly specialized legal assistant for Nepali law. Your **ONLY** source of information is the 'Context' provided below.

**CRITICAL RULES:**
1.  **NEVER** use any outside knowledge, personal knowledge, or information from your training data. Your knowledge is **STRICTLY LIMITED** to the provided text.
2.  Your answer **MUST** be based **100%** on the text found in the 'Context' (the numbered sources).
3.  **SYNTHESIS RULE:** If the user asks for a "definition" (like "What is 'cybercrime'?") and the text does not provide one, but *does* list examples (like "hacking," "data theft"), you MUST synthesize an answer.
    -   **Example Response:** "The law does not provide a single, formal definition for 'cybercrime'. However, it outlines several offenses considered computer-related crimes, such as unauthorized access [1], theft of source code [2], and publishing specific materials [3]."
4.  **FAILURE RULE:** If the 'Context' is **completely irrelevant** to the query (e.g., the query is about "cybercrime" and the context is only about "postal mail"), THEN and ONLY THEN must you respond with:
    "I am sorry, but the provided legal documents do not contain information on that topic."
5.  **CITATION RULE:** Cite every factual claim using the source number: [Number]. Place citations directly after the information they support: "The law states X [1]."
"""

    user_prompt = f"""**Context:**

{context}

**Question:** {query}

**Instructions:**
1.  Carefully read the 'Context' above.
2.  Answer the 'Question' using **ONLY** that context, following all rules (especially the SYNTHESIS RULE).
3.  If the context is completely irrelevant, respond with the required "I am sorry..." sentence.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0, # Set to 0.0 for maximum adherence to instructions
        max_tokens=1500
    )
    
    return response.choices[0].message.content

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Nepali Legal RAG API - Enhanced Version",
        "version": "3.0.4",
        "features": [
            "Query relevance filtering",
            "Bilingual query reformulation",
            "Advanced Synthesis Prompting",
            "Robust retrieval (no threshold)"
        ]
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
        metadata_fields = []
        if sample and sample['metadatas']:
            metadata_fields = list(sample['metadatas'][0].keys())
        
        return {
            "total_documents": count,
            "collection_name": "legal_documents",
            "model": "text-embedding-3-large",
            "version": "3.0.4",
            "available_metadata": metadata_fields
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # STEP 1: Check query relevance
        is_relevant, reason = check_query_relevance(request.query)
        
        if not is_relevant:
            return QueryResponse(
                query=request.query,
                query_variations=None,
                answer="I am sorry, but the provided legal documents do not contain information on that topic.",
                summary=None,
                sources=[],
                total_sources=0,
                out_of_context=True
            )
        
        # STEP 2: Search with reformulation (no threshold)
        n_results = min(request.n_results_per_query, 10)
        
        results, query_variations = search_with_reformulation(
            request.query, 
            n_results, 
            request.use_reformulation
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # STEP 3: Check if any documents were retrieved at all
        if not documents or len(documents) == 0:
            # This now only happens if the collection is empty or Chroma fails
            return QueryResponse(
                query=request.query,
                query_variations=query_variations if request.use_reformulation else None,
                answer="I am sorry, but the provided legal documents do not contain information on that topic.",
                summary=None,
                sources=[],
                total_sources=0,
                out_of_context=True
            )
        
        # STEP 4: Generate summary if requested
        summary = None
        if request.include_summary:
            summary = summarize_chunks(documents)
        
        # STEP 5: Generate answer (LLM does the filtering and synthesis)
        answer = generate_answer_with_citations(request.query, documents, metadatas)
        
        # STEP 6: Format sources
        sources = []
        for i, (meta, dist) in enumerate(zip(metadatas, distances), 1):
            page_nums = meta.get('page_numbers', [])
            if isinstance(page_nums, str):
                try:
                    page_nums = json.loads(page_nums)
                except:
                    page_nums = []
            if not isinstance(page_nums, list):
                page_nums = []
            
            sources.append(Source(
                index=i,
                filename=meta.get('filename', 'Unknown'),
                document_title=meta.get('document_title'),
                relevance=round(1 - dist, 3), # Distance is still useful to show
                volume=meta.get('volume', 'Unknown'),
                section_type=meta.get('section_type'),
                section_number=meta.get('section_number'),
                page_numbers=page_nums,
                citation=meta.get('citation')
            ))
        
        # Handle case where LLM *still* couldn't find answer in retrieved docs
        if answer == "I am sorry, but the provided legal documents do not contain information on that topic.":
             return QueryResponse(
                query=request.query,
                query_variations=query_variations if request.use_reformulation else None,
                answer=answer,
                summary=summary, # Show summary even if answer fails
                sources=[], # Return no sources if answer wasn't found in them
                total_sources=0,
                out_of_context=True
            )

        return QueryResponse(
            query=request.query,
            query_variations=query_variations if request.use_reformulation else None,
            answer=answer,
            summary=summary,
            sources=sources,
            total_sources=len(sources),
            out_of_context=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)