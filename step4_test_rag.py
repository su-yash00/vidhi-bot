# """
# STEP 4: Test RAG System
# """
# import os
# import chromadb
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# class SimpleRAG:
#     def __init__(self):
#         api_key = os.getenv('OPENAI_API_KEY')
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY not found!")
        
#         self.openai_client = OpenAI(api_key=api_key)
#         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
#         self.collection = self.chroma_client.get_collection(name="legal_documents")
#         print("✓ RAG system initialized")
    
#     def generate_query_embedding(self, query):
#         response = self.openai_client.embeddings.create(
#             model="text-embedding-3-large", 
#             input=query
#         )
#         return response.data[0].embedding
    
#     def search_documents(self, query, n_results=5):
#         print(f"\n🔍 Searching for: '{query}'")
#         query_embedding = self.generate_query_embedding(query)
#         results = self.collection.query(
#             query_embeddings=[query_embedding], 
#             n_results=n_results
#         )
#         return results
    
#     def generate_answer(self, query, context_documents):
#         context = "\n\n---\n\n".join([
#             f"[Document {i+1}]\n{doc}" 
#             for i, doc in enumerate(context_documents)
#         ])
        
#         system_prompt = """You are a helpful legal assistant for Nepali law.
# Answer questions based ONLY on the provided documents.
# Be precise and cite document numbers when referencing information."""

#         user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear answer with citations:"

#         response = self.openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.3,
#             max_tokens=1000
#         )
#         return response.choices[0].message.content
    
#     def query(self, question, n_results=3):
#         print("=" * 70)
#         results = self.search_documents(question, n_results)
        
#         documents = results['documents'][0]
#         metadatas = results['metadatas'][0]
#         distances = results['distances'][0]
        
#         print(f"✓ Found {len(documents)} relevant documents")
#         print("\n📚 Sources:")
#         for i, (meta, dist) in enumerate(zip(metadatas, distances)):
#             relevance = 1 - dist
#             print(f"   [{i+1}] {meta.get('filename', 'Unknown')} (relevance: {relevance:.2%})")
        
#         answer = self.generate_answer(question, documents)
        
#         print("\n💡 Answer:")
#         print("-" * 70)
#         print(answer)
#         print("-" * 70)

# def main():
#     print("=" * 70)
#     print("   STEP 4: TESTING RAG SYSTEM")
#     print("=" * 70)
    
#     try:
#         rag = SimpleRAG()
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return
    
#     # Test queries
#     test_queries = [
#         "What are the main provisions about telecommunications?",
#         "सञ्चार सम्बन्धी मुख्य कानूनहरू के हुन्?",
#     ]
    
#     for query in test_queries:
#         rag.query(query, n_results=3)
#         print("\n")
    
#     print("=" * 70)
#     print("   ✅ RAG SYSTEM IS WORKING!")
#     print("=" * 70)
#     print("\n🎯 Next: Build a web interface or API!")

# if __name__ == "__main__":
#     main()



# """
# STEP 4: Test RAG System with Google Gemini
# """
# import os
# import chromadb
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# class SimpleRAG:
#     def __init__(self):
#         api_key = os.getenv('GEMINI_API_KEY')
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY not found!")
        
#         genai.configure(api_key=api_key)
        
#         # Models
#         self.embedding_model = "models/text-embedding-004"
#         self.chat_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        
#         # ChromaDB
#         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
#         self.collection = self.chroma_client.get_collection(name="legal_documents")
        
#         print("✓ RAG system initialized with Gemini")
    
#     def generate_query_embedding(self, query):
#         """Generate embedding for query"""
#         result = genai.embed_content(
#             model=self.embedding_model,
#             content=query,
#             task_type="retrieval_query"
#         )
#         return result['embedding']
    
#     def search_documents(self, query, n_results=5):
#         """Search for relevant documents"""
#         print(f"\n🔍 Searching for: '{query}'")
#         query_embedding = self.generate_query_embedding(query)
        
#         results = self.collection.query(
#             query_embeddings=[query_embedding], 
#             n_results=n_results
#         )
#         return results
    
#     def generate_answer(self, query, context_documents):
#         """Generate answer using Gemini"""
#         context = "\n\n---\n\n".join([
#             f"[Document {i+1}]\n{doc}" 
#             for i, doc in enumerate(context_documents)
#         ])
        
#         prompt = f"""You are a helpful legal assistant for Nepali law.
# Answer questions based ONLY on the provided documents.
# Be precise and cite document numbers when referencing information.

# Context:
# {context}

# Question: {query}

# Provide a clear answer with citations:"""
        
#         response = self.chat_model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.3,
#                 max_output_tokens=1000,
#             )
#         )
        
#         return response.text
    
#     def query(self, question, n_results=3):
#         """Complete RAG query"""
#         print("=" * 70)
        
#         # Search
#         results = self.search_documents(question, n_results)
        
#         documents = results['documents'][0]
#         metadatas = results['metadatas'][0]
#         distances = results['distances'][0]
        
#         print(f"✓ Found {len(documents)} relevant documents")
        
#         # Display sources
#         print("\n📚 Sources:")
#         for i, (meta, dist) in enumerate(zip(metadatas, distances)):
#             relevance = 1 - dist
#             print(f"   [{i+1}] {meta.get('filename', 'Unknown')} (relevance: {relevance:.2%})")
        
#         # Generate answer
#         answer = self.generate_answer(question, documents)
        
#         print("\n💡 Answer:")
#         print("-" * 70)
#         print(answer)
#         print("-" * 70)


# def main():
#     print("=" * 70)
#     print("   STEP 4: TESTING RAG SYSTEM (GEMINI)")
#     print("=" * 70)
    
#     try:
#         rag = SimpleRAG()
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         print("\n📝 Make sure GEMINI_API_KEY is in your .env file")
#         return
    
#     # Test queries
#     test_queries = [
#         "What are the main provisions about telecommunications?",
#         "सञ्चार सम्बन्धी मुख्य कानूनहरू के हुन्?",
#     ]
    
#     for query in test_queries:
#         rag.query(query, n_results=3)
#         print("\n")
    
#     print("=" * 70)
#     print("   ✅ RAG SYSTEM IS WORKING!")
#     print("=" * 70)
#     print("\n🎯 Next: Build a web interface or API!")


# if __name__ == "__main__":
#     main()

"""
STEP 4: Test RAG System with Google Gemini
"""
import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class SimpleRAG:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found!")
        
        genai.configure(api_key=api_key)
        
        # Models - Use the ones available in your API
        self.embedding_model = "models/text-embedding-004"
        self.chat_model = genai.GenerativeModel('models/gemini-2.5-flash')  # Fast and available
        
        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.chroma_client.get_collection(name="legal_documents")
        
        print("✓ RAG system initialized with Gemini")
    
    def generate_query_embedding(self, query):
        """Generate embedding for query"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def search_documents(self, query, n_results=5):
        """Search for relevant documents"""
        print(f"\n🔍 Searching for: '{query}'")
        query_embedding = self.generate_query_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results
        )
        return results
    
    def generate_answer(self, query, context_documents):
        """Generate answer using Gemini"""
        context = "\n\n---\n\n".join([
            f"[Document {i+1}]\n{doc}" 
            for i, doc in enumerate(context_documents)
        ])
        
        prompt = f"""You are a helpful legal assistant for Nepali law.
Answer questions based ONLY on the provided documents.
Be precise and cite document numbers when referencing information.

Context:
{context}

Question: {query}

Provide a clear answer with citations:"""
        
        response = self.chat_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        
        return response.text
    
    def query(self, question, n_results=3):
        """Complete RAG query"""
        print("=" * 70)
        
        # Search
        results = self.search_documents(question, n_results)
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        print(f"✓ Found {len(documents)} relevant documents")
        
        # Display sources
        print("\n📚 Sources:")
        for i, (meta, dist) in enumerate(zip(metadatas, distances)):
            relevance = 1 - dist
            print(f"   [{i+1}] {meta.get('filename', 'Unknown')} (relevance: {relevance:.2%})")
        
        # Generate answer
        answer = self.generate_answer(question, documents)
        
        print("\n💡 Answer:")
        print("-" * 70)
        print(answer)
        print("-" * 70)


def main():
    print("=" * 70)
    print("   STEP 4: TESTING RAG SYSTEM (GEMINI)")
    print("=" * 70)
    
    try:
        rag = SimpleRAG()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📝 Make sure GEMINI_API_KEY is in your .env file")
        return
    
    # Test queries
    test_queries = [
        "What are the main provisions about telecommunications?",
        "सञ्चार सम्बन्धी मुख्य कानूनहरू के हुन्?",
    ]
    
    for query in test_queries:
        rag.query(query, n_results=3)
        print("\n")
    
    print("=" * 70)
    print("   ✅ RAG SYSTEM IS WORKING!")
    print("=" * 70)
    print("\n🎯 Next: Build a web interface or API!")


if __name__ == "__main__":
    main()