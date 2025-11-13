# # """
# # STEP 2: Generate Embeddings (FIXED)
# # """

# # import json
# # import os
# # from pathlib import Path
# # from openai import OpenAI
# # from tqdm import tqdm
# # import time
# # from dotenv import load_dotenv

# # load_dotenv()

# # class EmbeddingGenerator:
# #     def __init__(self):
# #         api_key = os.getenv('OPENAI_API_KEY')
# #         if not api_key:
# #             raise ValueError("❌ OPENAI_API_KEY not found in .env file!")
        
# #         # Set the key in environment (new client reads it from env)
# #         os.environ["OPENAI_API_KEY"] = api_key
        
# #         self.client = OpenAI()
# #         self.model = "text-embedding-3-large"
    
# #     def truncate_text(self, text, max_tokens=8000):
# #         """Truncate text to fit within token limit"""
# #         # Rough estimate: 1 token ≈ 0.75 words
# #         max_words = int(max_tokens * 0.75)
# #         words = text.split()
# #         if len(words) > max_words:
# #             return ' '.join(words[:max_words])
# #         return text
    
# #     def generate_single_embedding(self, text):
# #         """Generate embedding for a single text"""
# #         try:
# #             # Truncate if needed
# #             text = self.truncate_text(text)
            
# #             response = self.client.embeddings.create(
# #                 model=self.model,
# #                 input=text
# #             )
# #             return response.data[0].embedding
# #         except Exception as e:
# #             print(f"❌ Error: {e}")
# #             return [0] * 3072
    
# #     def generate_batch_embeddings(self, texts):
# #         """Generate embeddings one by one (safer)"""
# #         embeddings = []
        
# #         print(f"Generating embeddings for {len(texts)} chunks...")
        
# #         for i, text in enumerate(tqdm(texts, desc="Processing chunks")):
# #             try:
# #                 embedding = self.generate_single_embedding(text)
# #                 embeddings.append(embedding)
                
# #                 # Rate limiting
# #                 if i % 10 == 0:
# #                     time.sleep(0.5)
# #                 else:
# #                     time.sleep(0.1)
                    
# #             except Exception as e:
# #                 print(f"\n❌ Failed on chunk {i}: {e}")
# #                 embeddings.append([0] * 3072)
        
# #         return embeddings
    
# #     def process_chunks(self, chunks_file, output_file):
# #         """Process all chunks and generate embeddings"""
# #         print(f"📖 Loading chunks from: {chunks_file}")
# #         with open(chunks_file, 'r', encoding='utf-8') as f:
# #             chunks = json.load(f)
        
# #         print(f"   Found {len(chunks)} chunks")
        
# #         # Extract texts
# #         texts = [chunk['text'] for chunk in chunks]
        
# #         # Calculate cost estimate (more accurate now)
# #         total_words = sum(len(text.split()) for text in texts)
# #         total_tokens = int(total_words * 1.3)
# #         cost_estimate = (total_tokens / 1_000_000) * 0.13
# #         print(f"\n💰 Estimated cost: ${cost_estimate:.2f}")
# #         print(f"   Total tokens (approx): {total_tokens:,}")
        
# #         # Ask for confirmation
# #         response = input("\nProceed with embedding generation? (yes/no): ")
# #         if response.lower() not in ['yes', 'y']:
# #             print("❌ Cancelled")
# #             return
        
# #         # Generate embeddings
# #         print(f"\n🔄 Generating embeddings...")
# #         embeddings = self.generate_batch_embeddings(texts)
        
# #         # Combine chunks with embeddings
# #         print(f"\n💾 Saving embedded chunks...")
# #         embedded_chunks = []
# #         for chunk, embedding in zip(chunks, embeddings):
# #             embedded_chunks.append({
# #                 'text': chunk['text'],
# #                 'embedding': embedding,
# #                 'metadata': chunk['metadata']
# #             })
        
# #         # Save to file
# #         output_file.parent.mkdir(exist_ok=True)
# #         with open(output_file, 'w', encoding='utf-8') as f:
# #             json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        
# #         print(f"✓ Saved {len(embedded_chunks)} embedded chunks to: {output_file}")
# #         print(f"✓ Actual cost: ~${cost_estimate:.2f}")


# # def main():
# #     print("=" * 70)
# #     print("   STEP 2: GENERATING EMBEDDINGS (FIXED VERSION)")
# #     print("=" * 70)
    
# #     chunks_file = Path("data/processed/all_chunks.json")
# #     if not chunks_file.exists():
# #         print("❌ Chunks file not found!")
# #         return
    
# #     try:
# #         generator = EmbeddingGenerator()
# #     except ValueError as e:
# #         print(e)
# #         print("\n📝 Create a .env file with: OPENAI_API_KEY=sk-your-key")
# #         return
    
# #     output_file = Path("data/embeddings/embedded_chunks.json")
# #     generator.process_chunks(chunks_file, output_file)
    
# #     print("\n" + "=" * 70)
# #     print("   EMBEDDING GENERATION COMPLETE!")
# #     print("=" * 70)
# #     print("\n🎯 Next step: Run step3_build_vector_store.py")


# # if __name__ == "__main__":
# #     main()


# """
# STEP 5: RAG System Evaluation Metrics (Gemini Version)
# Measures: Retrieval Quality, Generation Accuracy, Relevance, Latency
# """

# import json
# import time
# import chromadb
# import google.generativeai as genai
# from pathlib import Path
# from typing import List, Dict
# import numpy as np
# from dotenv import load_dotenv
# import os
# from datetime import datetime

# load_dotenv()

# class RAGEvaluator:
#     def __init__(self):
#         self.api_key = os.getenv('GEMINI_API_KEY')
#         if not self.api_key:
#             raise ValueError("GEMINI_API_KEY not found!")
        
#         genai.configure(api_key=self.api_key)
        
#         self.embedding_model = "models/text-embedding-004"
#         self.chat_model = genai.GenerativeModel('gemini-1.5-flash')
        
#         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
#         self.collection = self.chroma_client.get_collection(name="legal_documents")
        
#         self.metrics = {
#             'retrieval': {},
#             'generation': {},
#             'relevance': {},
#             'latency': {}
#         }
    
#     def _generate_embedding(self, text: str):
#         """Generate embedding using Gemini"""
#         result = genai.embed_content(
#             model=self.embedding_model,
#             content=text,
#             task_type="retrieval_query"
#         )
#         return result['embedding']
    
#     def _generate_answer(self, query: str, documents: List[str]) -> str:
#         """Generate answer using Gemini"""
#         context = "\n\n---\n\n".join([f"[Source {i+1}]\n{doc}" for i, doc in enumerate(documents)])
        
#         prompt = f"""You are a legal assistant. Always cite sources.

# Context:
# {context}

# Question: {query}

# Answer:"""
        
#         response = self.chat_model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.2,
#                 max_output_tokens=500,
#             )
#         )
        
#         return response.text
    
#     # RETRIEVAL METRICS
#     def calculate_retrieval_metrics(self, test_queries: List[Dict]):
#         print("\n" + "="*70)
#         print("   RETRIEVAL QUALITY METRICS")
#         print("="*70)
        
#         precisions = []
#         recalls = []
#         mrr_scores = []
        
#         for test_case in test_queries:
#             query = test_case['query']
#             relevant_docs = test_case.get('relevant_docs', [])
            
#             query_embedding = self._generate_embedding(query)
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=10
#             )
            
#             retrieved_filenames = [doc.get('filename', '') for doc in results['metadatas'][0]]
            
#             if relevant_docs:
#                 precision = len(set(retrieved_filenames) & set(relevant_docs)) / len(retrieved_filenames)
#                 recall = len(set(retrieved_filenames) & set(relevant_docs)) / len(relevant_docs)
                
#                 mrr = 0
#                 for i, doc in enumerate(retrieved_filenames, 1):
#                     if doc in relevant_docs:
#                         mrr = 1.0 / i
#                         break
                
#                 precisions.append(precision)
#                 recalls.append(recall)
#                 mrr_scores.append(mrr)
            
#             time.sleep(0.2)  # Rate limiting
        
#         self.metrics['retrieval'] = {
#             'precision@10': np.mean(precisions) if precisions else 0,
#             'recall@10': np.mean(recalls) if recalls else 0,
#             'mrr': np.mean(mrr_scores) if mrr_scores else 0,
#             'num_test_queries': len(test_queries)
#         }
        
#         print(f"\n📊 Retrieval Quality:")
#         print(f"   Precision@10: {self.metrics['retrieval']['precision@10']:.3f} ({self.metrics['retrieval']['precision@10']*100:.1f}%)")
#         print(f"   Recall@10:    {self.metrics['retrieval']['recall@10']:.3f} ({self.metrics['retrieval']['recall@10']*100:.1f}%)")
#         print(f"   MRR:          {self.metrics['retrieval']['mrr']:.3f}")
    
#     # GENERATION METRICS
#     def calculate_generation_metrics(self, test_queries: List[Dict]):
#         print("\n" + "="*70)
#         print("   GENERATION ACCURACY METRICS")
#         print("="*70)
        
#         faithfulness_scores = []
#         citation_counts = []
        
#         for test_case in test_queries:
#             query = test_case['query']
            
#             query_embedding = self._generate_embedding(query)
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=5
#             )
            
#             documents = results['documents'][0]
#             answer = self._generate_answer(query, documents)
            
#             # Check citations
#             import re
#             citations = re.findall(r'\[Source \d+\]', answer)
#             citation_counts.append(len(citations))
            
#             # Simple faithfulness check
#             has_citations = len(citations) > 0
#             is_reasonable_length = 50 < len(answer.split()) < 500
#             faithfulness = 1.0 if (has_citations and is_reasonable_length) else 0.5
#             faithfulness_scores.append(faithfulness)
            
#             time.sleep(0.5)  # Rate limiting
        
#         self.metrics['generation'] = {
#             'faithfulness': np.mean(faithfulness_scores),
#             'avg_citations': np.mean(citation_counts),
#             'num_queries': len(test_queries)
#         }
        
#         print(f"\n🎯 Generation Quality:")
#         print(f"   Faithfulness:   {self.metrics['generation']['faithfulness']:.3f} ({self.metrics['generation']['faithfulness']*100:.1f}%)")
#         print(f"   Avg Citations:  {self.metrics['generation']['avg_citations']:.1f} per answer")
    
#     # RELEVANCE METRICS
#     def calculate_relevance_metrics(self, test_queries: List[Dict]):
#         print("\n" + "="*70)
#         print("   RELEVANCE & USER SATISFACTION")
#         print("="*70)
        
#         satisfaction_scores = []
        
#         for test_case in test_queries:
#             query = test_case['query']
            
#             query_embedding = self._generate_embedding(query)
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=5
#             )
            
#             documents = results['documents'][0]
#             distances = results['distances'][0]
#             answer = self._generate_answer(query, documents)
            
#             # Calculate satisfaction score
#             avg_relevance = 1 - np.mean(distances)
#             has_citations = '[Source' in answer
#             answer_length = len(answer.split())
            
#             score = avg_relevance * 0.5
#             if has_citations:
#                 score += 0.3
#             if 50 <= answer_length <= 500:
#                 score += 0.2
            
#             satisfaction_scores.append(min(score, 1.0))
#             time.sleep(0.5)  # Rate limiting
        
#         self.metrics['relevance'] = {
#             'avg_relevance': np.mean([1 - d for result in [self.collection.query(query_embeddings=[self._generate_embedding(tq['query'])], n_results=5) for tq in test_queries] for d in result['distances'][0]]),
#             'user_satisfaction': np.mean(satisfaction_scores),
#             'satisfaction_rate': sum(1 for s in satisfaction_scores if s >= 0.7) / len(satisfaction_scores)
#         }
        
#         print(f"\n⭐ Relevance & Satisfaction:")
#         print(f"   Avg Relevance:      {self.metrics['relevance']['avg_relevance']:.3f} ({self.metrics['relevance']['avg_relevance']*100:.1f}%)")
#         print(f"   User Satisfaction:  {self.metrics['relevance']['user_satisfaction']:.3f} ({self.metrics['relevance']['user_satisfaction']*100:.1f}%)")
#         print(f"   Satisfaction Rate:  {self.metrics['relevance']['satisfaction_rate']:.3f} ({self.metrics['relevance']['satisfaction_rate']*100:.1f}%)")
    
#     # LATENCY METRICS
#     def calculate_latency_metrics(self, test_queries: List[Dict]):
#         print("\n" + "="*70)
#         print("   LATENCY & EFFICIENCY METRICS")
#         print("="*70)
        
#         embedding_times = []
#         retrieval_times = []
#         generation_times = []
#         total_times = []
        
#         for test_case in test_queries:
#             query = test_case['query']
            
#             total_start = time.time()
            
#             # Embedding
#             embed_start = time.time()
#             query_embedding = self._generate_embedding(query)
#             embedding_times.append(time.time() - embed_start)
            
#             # Retrieval
#             retrieval_start = time.time()
#             results = self.collection.query(query_embeddings=[query_embedding], n_results=5)
#             retrieval_times.append(time.time() - retrieval_start)
            
#             # Generation
#             gen_start = time.time()
#             documents = results['documents'][0]
#             answer = self._generate_answer(query, documents)
#             generation_times.append(time.time() - gen_start)
            
#             total_times.append(time.time() - total_start)
#             time.sleep(0.5)  # Rate limiting
        
#         self.metrics['latency'] = {
#             'avg_embedding_time': np.mean(embedding_times),
#             'avg_retrieval_time': np.mean(retrieval_times),
#             'avg_generation_time': np.mean(generation_times),
#             'avg_total_time': np.mean(total_times),
#             'p95_total_time': np.percentile(total_times, 95),
#             'p99_total_time': np.percentile(total_times, 99)
#         }
        
#         print(f"\n⚡ Latency:")
#         print(f"   Avg Embedding:   {self.metrics['latency']['avg_embedding_time']*1000:.0f}ms")
#         print(f"   Avg Retrieval:   {self.metrics['latency']['avg_retrieval_time']*1000:.0f}ms")
#         print(f"   Avg Generation:  {self.metrics['latency']['avg_generation_time']*1000:.0f}ms")
#         print(f"   Avg Total:       {self.metrics['latency']['avg_total_time']*1000:.0f}ms ({self.metrics['latency']['avg_total_time']:.2f}s)")
#         print(f"   P95 Total:       {self.metrics['latency']['p95_total_time']*1000:.0f}ms")
    
#     def save_metrics(self, filepath: str = "evaluation_results_gemini.json"):
#         results = {
#             'timestamp': datetime.now().isoformat(),
#             'ai_provider': 'Google Gemini',
#             'models': {
#                 'embeddings': 'text-embedding-004',
#                 'chat': 'gemini-1.5-flash'
#             },
#             'metrics': self.metrics
#         }
        
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
        
#         print(f"\n💾 Metrics saved to: {filepath}")

# def create_test_queries():
#     return [
#         {
#             'query': 'What are telecommunications laws?',
#             'relevant_docs': ['telecommunications-act-2053.pdf'],
#         },
#         {
#             'query': 'नागरिकता कानून के हो?',
#             'relevant_docs': ['citizenship-act.pdf'],
#         },
#         {
#             'query': 'What are property rights?',
#             'relevant_docs': ['property-act.pdf'],
#         },
#         {
#             'query': 'सञ्चार सम्बन्धी कानून',
#             'relevant_docs': ['telecommunications-act.pdf'],
#         },
#         {
#             'query': 'media regulation laws',
#             'relevant_docs': ['media-regulation.pdf'],
#         }
#     ]

# def main():
#     print("="*70)
#     print("   RAG SYSTEM EVALUATION (GEMINI)")
#     print("="*70)
    
#     try:
#         evaluator = RAGEvaluator()
#     except ValueError as e:
#         print(f"❌ {e}")
#         print("\n📝 Create a .env file with: GEMINI_API_KEY=your-key-here")
#         return
    
#     test_queries = create_test_queries()
    
#     print(f"\n📝 Running evaluation with {len(test_queries)} test queries...")
    
#     evaluator.calculate_retrieval_metrics(test_queries)
#     evaluator.calculate_generation_metrics(test_queries)
#     evaluator.calculate_relevance_metrics(test_queries)
#     evaluator.calculate_latency_metrics(test_queries)
    
#     evaluator.save_metrics("evaluation_results_gemini.json")
    
#     print("\n" + "="*70)
#     print("   EVALUATION SUMMARY")
#     print("="*70)
#     print(f"   Retrieval Quality:  {evaluator.metrics['retrieval']['precision@10']*100:.1f}%")
#     print(f"   Generation Quality: {evaluator.metrics['generation']['faithfulness']*100:.1f}%")
#     print(f"   User Satisfaction:  {evaluator.metrics['relevance']['user_satisfaction']*100:.1f}%")
#     print(f"   Avg Response Time:  {evaluator.metrics['latency']['avg_total_time']*1000:.0f}ms")
#     print("\n✅ Evaluation complete!")

# if __name__ == "__main__":
#     main()



"""
STEP 2: Generate Embeddings with Google Gemini API
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import time
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
class EmbeddingGenerator:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        print("GEMINI API KEY", api_key)
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file!")
        
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"
        
        print(f"✓ Initialized Gemini embeddings: {self.model}")
    
    def truncate_text(self, text, max_chars=20000):
        """Truncate text to fit within character limit"""
        # Gemini has a character limit, not token limit
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def generate_single_embedding(self, text):
        """Generate embedding for a single text"""
        try:
            # Truncate if needed
            text = self.truncate_text(text)
            
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"  # Use "retrieval_document" for indexing
            )
            return result['embedding']
        except Exception as e:
            print(f"❌ Error: {e}")
            # Return zero vector (768 dims for Gemini)
            return [0] * 768
    
    def generate_batch_embeddings(self, texts):
        """Generate embeddings one by one (safer)"""
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        for i, text in enumerate(tqdm(texts, desc="Processing chunks")):
            try:
                embedding = self.generate_single_embedding(text)
                embeddings.append(embedding)
                
                # Rate limiting - Gemini free tier: 1,500 req/day
                # ~10 requests per second is safe
                if i % 10 == 0:
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"\n❌ Failed on chunk {i}: {e}")
                embeddings.append([0] * 768)
        
        return embeddings
    
    def process_chunks(self, chunks_file, output_file):
        """Process all chunks and generate embeddings"""
        print(f"📖 Loading chunks from: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"   Found {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Cost estimate (Gemini is FREE for standard usage)
        total_chars = sum(len(text) for text in texts)
        total_words = sum(len(text.split()) for text in texts)
        
        print(f"\n💰 Cost: FREE (Gemini API)")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Total words (approx): {total_words:,}")
        print(f"   Rate limit: 1,500 requests/day")
        print(f"   Estimated time: ~{len(texts) * 0.15 / 60:.1f} minutes")
        
        # Ask for confirmation
        response = input("\nProceed with embedding generation? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled")
            return
        
        # Generate embeddings
        print(f"\n🔄 Generating embeddings with Gemini...")
        embeddings = self.generate_batch_embeddings(texts)
        
        # Combine chunks with embeddings
        print(f"\n💾 Saving embedded chunks...")
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append({
                'text': chunk['text'],
                'embedding': embedding,
                'metadata': chunk['metadata']
            })
        
        # Save to file
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(embedded_chunks)} embedded chunks to: {output_file}")
        print(f"✓ Embedding dimensions: 768 (Gemini text-embedding-004)")
        print(f"✓ Cost: $0.00 (FREE)")


def main():
    print("=" * 70)
    print("   STEP 2: GENERATING EMBEDDINGS (GEMINI VERSION)")
    print("=" * 70)
    
    chunks_file = Path("data/processed/all_chunks.json")
    if not chunks_file.exists():
        print("❌ Chunks file not found!")
        print("   Make sure you've run step1_process_pdfs.py first")
        return
    
    try:
        generator = EmbeddingGenerator()
    except ValueError as e:
        print(e)
        print("\n📝 Create a .env file with: GEMINI_API_KEY=your-key-here")
        print("   Get your key at: https://makersuite.google.com/app/apikey")
        return
    
    output_file = Path("data/embeddings/embedded_chunks.json")
    generator.process_chunks(chunks_file, output_file)
    
    print("\n" + "=" * 70)
    print("   EMBEDDING GENERATION COMPLETE!")
    print("=" * 70)
    print("\n🎯 Next step: Run step3_build_vector_store.py")


if __name__ == "__main__":
    main()