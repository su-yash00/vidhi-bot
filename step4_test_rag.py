"""
STEP 4: Enhanced RAG System with Query Reformulation and Better Citations
"""
import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class EnhancedRAG:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found!")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.chroma_client.get_collection(name="legal_documents")
        print("✓ Enhanced RAG system initialized")

    def reformulate_query(self, original_query: str) -> List[str]:
        """
        Generate multiple query variations for better retrieval
        """
        prompt = f"""Given this legal query, generate 3 different variations that capture the same intent.
Consider:
- Technical legal terminology
- Nepali and English variants
- Different phrasings of the same question

Original query: {original_query}

Provide ONLY the 3 variations, one per line, without numbering or explanation."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a legal query reformulation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            return [original_query] + variations[:3]
        except:
            return [original_query]

    def generate_query_embedding(self, query: str):
        """Generate embedding for a single query"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        return response.data[0].embedding

    def search_with_reformulation(self, query: str, n_results_per_query: int = 3) -> Dict:
        """
        Search using multiple query variations and combine results
        """
        print(f"\n🔍 Original query: '{query}'")
        
        # Generate query variations
        query_variations = self.reformulate_query(query)
        print(f"   Generated {len(query_variations)} query variations")
        
        # Search with each variation
        all_results = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }
        
        seen_ids = set()
        
        for i, q_var in enumerate(query_variations):
            query_embedding = self.generate_query_embedding(q_var)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results_per_query
            )
            
            # Combine results, avoiding duplicates
            for j in range(len(results['ids'][0])):
                doc_id = results['ids'][0][j]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results['documents'].append(results['documents'][0][j])
                    all_results['metadatas'].append(results['metadatas'][0][j])
                    # Adjust distance based on query variation (original query gets priority)
                    adjusted_distance = results['distances'][0][j] * (1 + i * 0.1)
                    all_results['distances'].append(adjusted_distance)
                    all_results['ids'].append(doc_id)
        
        # Sort by adjusted distance and limit
        sorted_indices = sorted(range(len(all_results['distances'])), 
                               key=lambda i: all_results['distances'][i])
        
        max_results = min(10, len(sorted_indices))
        sorted_indices = sorted_indices[:max_results]
        
        # Reorder all results
        final_results = {
            'documents': [[all_results['documents'][i] for i in sorted_indices]],
            'metadatas': [[all_results['metadatas'][i] for i in sorted_indices]],
            'distances': [[all_results['distances'][i] for i in sorted_indices]],
            'ids': [[all_results['ids'][i] for i in sorted_indices]]
        }
        
        return final_results

    def extract_relevant_quotes(self, query: str, documents: List[str]) -> List[str]:
        """
        Extract most relevant quotes from documents
        """
        quotes = []
        for doc in documents[:5]:  # Top 5 docs
            # Split into sentences
            sentences = doc.split('.')
            # Get sentences that are substantial
            good_sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 50]
            if good_sentences:
                quotes.append(good_sentences[0])  # First substantial sentence
        
        return quotes

    def summarize_chunks(self, documents: List[str], metadatas: List[dict]) -> str:
        """
        Create a concise summary of retrieved chunks
        """
        combined_text = "\n\n".join(documents[:5])  # Top 5 chunks
        
        prompt = f"""Summarize the key legal points from these document excerpts in 3-4 bullet points:

{combined_text}

Provide a concise summary highlighting the main legal provisions."""

        try:
            response = self.openai_client.chat.completions.create(
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
            return "Summary unavailable."

    def generate_answer_with_citations(self, query: str, documents: List[str], 
                                       metadatas: List[dict], distances: List[float]):
        """
        Generate comprehensive answer with inline citations and references
        """
        # Build context with citation markers
        context_parts = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            citation_parts = [f"[{i+1}]"]
            
            # Document title
            if meta.get('document_title'):
                citation_parts.append(meta['document_title'][:50])
            
            # Section info
            if meta.get('section_type') and meta.get('section_number'):
                section_type = meta['section_type'].replace('_nepali', '').title()
                citation_parts.append(f"{section_type} {meta['section_number']}")
            
            # Page numbers
            if meta.get('page_numbers'):
                pages = meta['page_numbers']
                if isinstance(pages, str):
                    try:
                        pages = json.loads(pages)
                    except:
                        pages = []
                
                if pages and len(pages) > 0:
                    if len(pages) == 1:
                        citation_parts.append(f"p.{pages[0]}")
                    else:
                        citation_parts.append(f"pp.{min(pages)}-{max(pages)}")
            
            marker = " ".join(citation_parts)
            context_parts.append(f"{marker}\n{doc}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Extract relevant quotes
        quotes = self.extract_relevant_quotes(query, documents)
        quotes_section = "\n".join([f"• \"{q}\"" for q in quotes[:3]])
        
        system_prompt = """You are an expert legal assistant for Nepali law. Provide accurate, well-cited answers.

CRITICAL CITATION RULES:
1. ALWAYS cite sources using [Number] format
2. Place citations immediately after claims: "The law states X [1]"
3. Use multiple citations when combining info: "X [1] and Y [2]"
4. Every factual claim MUST have a citation
5. Be precise and comprehensive
6. Start with a direct answer, then provide details
7. End with a summary of key points

Examples:
✓ "Telecommunications require licensing [1]."
✓ "Property rights are protected [2, 3] with specific provisions in Section 5 [2]."
✓ "According to Article 12 [4], citizens have the right to..."

Answer based ONLY on provided documents."""

        user_prompt = f"""Context from legal documents:

{context}

Relevant quotes:
{quotes_section}

Question: {query}

Provide a comprehensive answer with:
1. Direct answer to the question
2. Supporting details with citations
3. Relevant legal provisions
4. Summary of key points

Use inline citations [1], [2], etc. for every claim."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        return response.choices[0].message.content

    def format_source(self, metadata: dict, distance: float, index: int) -> str:
        """Format a source reference nicely"""
        parts = [f"[{index}]"]
        
        relevance = 1 - distance
        parts.append(f"({relevance:.1%} relevant)")
        
        if metadata.get('document_title'):
            parts.append(f"\n    Title: {metadata['document_title']}")
        
        parts.append(f"\n    File: {metadata.get('filename', 'Unknown')}")
        
        if metadata.get('section_type') and metadata.get('section_number'):
            section = metadata['section_type'].replace('_nepali', '').title()
            parts.append(f"\n    Section: {section} {metadata['section_number']}")
        
        if metadata.get('page_numbers'):
            pages = metadata['page_numbers']
            if isinstance(pages, str):
                try:
                    pages = json.loads(pages)
                except:
                    pages = []
            if pages:
                if len(pages) == 1:
                    parts.append(f"\n    Page: {pages[0]}")
                else:
                    parts.append(f"\n    Pages: {min(pages)}-{max(pages)}")
        
        parts.append(f"\n    Volume: {metadata.get('volume', 'Unknown')}")
        
        return " ".join(parts)

    def query(self, question: str, n_results_per_query: int = 3):
        """
        Enhanced query with reformulation and comprehensive response
        """
        print("=" * 70)
        
        # Search with query reformulation
        results = self.search_with_reformulation(question, n_results_per_query)
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        print(f"\n✓ Found {len(documents)} unique relevant documents")
        
        # Generate summary of chunks
        print("\n📋 Generating summary of retrieved content...")
        summary = self.summarize_chunks(documents, metadatas)
        print("\n" + "─" * 70)
        print("SUMMARY OF RETRIEVED CONTENT:")
        print(summary)
        print("─" * 70)
        
        # Generate answer
        print("\n🤖 Generating comprehensive answer...")
        answer = self.generate_answer_with_citations(question, documents, metadatas, distances)
        
        print("\n💡 ANSWER:")
        print("═" * 70)
        print(answer)
        print("═" * 70)
        
        # Show sources
        print("\n📚 SOURCES:")
        print("─" * 70)
        for i, (meta, dist) in enumerate(zip(metadatas[:10], distances[:10]), 1):
            print(self.format_source(meta, dist, i))
            print()
        
        return {
            'query': question,
            'answer': answer,
            'summary': summary,
            'sources': [
                {
                    'index': i,
                    'metadata': meta,
                    'relevance': 1 - dist
                }
                for i, (meta, dist) in enumerate(zip(metadatas, distances), 1)
            ]
        }


def main():
    print("=" * 70)
    print("   STEP 4: TESTING ENHANCED RAG SYSTEM")
    print("=" * 70)
    
    try:
        rag = EnhancedRAG()
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test queries
    test_queries = [
        "What are the main provisions about telecommunications licensing?",
        "सञ्चार सम्बन्धी मुख्य कानूनहरू के हुन्?",
        "What are the penalties for violating telecom regulations?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'='*70}")
        result = rag.query(query, n_results_per_query=3)
        
        if i < len(test_queries):
            input("\nPress Enter to continue to next query...")
    
    print("\n" + "=" * 70)
    print("   ✅ ENHANCED RAG SYSTEM IS WORKING!")
    print("=" * 70)
    print("\n✨ Improvements implemented:")
    print("   ✓ Query reformulation for better retrieval")
    print("   ✓ Chunk summarization")
    print("   ✓ Enhanced citations with titles, sections, pages")
    print("   ✓ Relevant quote extraction")
    print("   ✓ Comprehensive source formatting")
    print("\n🎯 Next: Run step5_fastapi_backend_enhanced.py")


if __name__ == "__main__":
    main()