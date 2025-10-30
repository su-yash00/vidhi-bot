"""
STEP 5: Debug & Test RAG System
This script tests the Retrieval-Augmented Generation system with full debug info.
"""

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DebugRAG:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        self.openai_client = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.chroma_client.get_collection("legal_documents")
        print("✅ RAG system initialized successfully!\n")

    def generate_query_embedding(self, query):
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        embedding = response.data[0].embedding
        print(f"[DEBUG] Query embedding length: {len(embedding)}")
        return embedding

    def search_documents(self, query, n_results=5):
        print(f"\n🔍 Searching for: '{query}'")
        query_embedding = self.generate_query_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def debug_search(self, query, n_results=5):
        results = self.search_documents(query, n_results)
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        similarities = [1 - d for d in distances]
        for i, (doc, meta, sim) in enumerate(zip(documents, metadatas, similarities)):
            print(f"\n[Document {i+1}] Relevance: {sim:.2%}")
            print(f"Filename: {meta.get('filename', 'Unknown.pdf')}")
            print(f"Text snippet: {doc[:300]}...")  # show first 300 chars

        return documents, metadatas, similarities

    def generate_answer(self, query, context_documents):
        context = "\n\n---\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(context_documents)])
        system_prompt = (
            "You are a helpful legal assistant for Nepali law. "
            "Answer questions based ONLY on the provided documents. "
            "Be precise and cite document numbers when referencing information."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear answer with citations:"

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def query(self, question, n_results=5, relevance_threshold=0.5):
        documents, metadatas, similarities = self.debug_search(question, n_results)

        # Filter by relevance
        relevant = [(doc, meta, sim) for doc, meta, sim in zip(documents, metadatas, similarities) if sim >= relevance_threshold]

        if not relevant:
            print("\n⚠️ No document passed the threshold. Showing top 3 closest matches anyway.")
            # fallback to top 3 regardless of threshold
            relevant = list(zip(documents[:3], metadatas[:3], similarities[:3]))

        relevant.sort(key=lambda x: x[2], reverse=True)
        relevant_docs, relevant_metas, relevant_sims = zip(*relevant)

        print(f"\n✓ Selected {len(relevant_docs)} documents for answer generation:\n")
        for i, (meta, sim) in enumerate(zip(relevant_metas, relevant_sims)):
            print(f"[{i+1}] {meta.get('filename', 'Unknown.pdf')} - relevance: {sim:.2%}")

        answer = self.generate_answer(question, relevant_docs)
        print("\nAnswer:")
        print("-" * 70)
        print(answer)
        print("-" * 70)

def main():
    rag = DebugRAG()

    test_queries = [
        "रेडियो ऐन २०१४ अनुसार रेडियो स्टेशनको अनुमति कसरी लिइन्छ?",
        "सञ्चार सम्बन्धी मुख्य कानूनहरू के हुन्?",
        "What does the Radio Act 2014 say about radio station licenses?","hiiiiibjhfcfg"
    ]

    for q in test_queries:
        rag.query(q)
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
