"""
STEP 5: RAG System Evaluation Metrics (Gemini Version)
Measures: Retrieval Quality, Generation Accuracy, Relevance, Latency
"""

import json
import time
import chromadb
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

class RAGEvaluator:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found!")
        
        genai.configure(api_key=self.api_key)
        
        self.embedding_model = "models/text-embedding-004"
        self.chat_model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.chroma_client.get_collection(name="legal_documents")
        
        self.metrics = {
            'retrieval': {},
            'generation': {},
            'relevance': {},
            'latency': {}
        }
    
    def _generate_embedding(self, text: str):
        """Generate embedding using Gemini"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def _generate_answer(self, query: str, documents: List[str]) -> str:
        """Generate answer using Gemini with safety handling"""
        context = "\n\n---\n\n".join([f"[Source {i+1}]\n{doc}" for i, doc in enumerate(documents)])
        
        prompt = f"""You are a legal assistant. Always cite sources.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.chat_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                ),
                safety_settings={
                    'HARASSMENT': 'block_none',
                    'HATE_SPEECH': 'block_none',
                    'SEXUALLY_EXPLICIT': 'block_none',
                    'DANGEROUS_CONTENT': 'block_none'
                }
            )
            
            # Check if response has text
            if response.text:
                return response.text
            else:
                # If blocked, return a message about it
                return f"[Response blocked - finish_reason: {response.candidates[0].finish_reason}]"
                
        except Exception as e:
            return f"[Error generating answer: {str(e)[:100]}]"
    
    # RETRIEVAL METRICS
    def calculate_retrieval_metrics(self, test_queries: List[Dict]):
        print("\n" + "="*70)
        print("   RETRIEVAL QUALITY METRICS")
        print("="*70)
        
        precisions = []
        recalls = []
        mrr_scores = []
        
        for test_case in test_queries:
            query = test_case['query']
            relevant_docs = test_case.get('relevant_docs', [])
            
            query_embedding = self._generate_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )
            
            retrieved_filenames = [doc.get('filename', '') for doc in results['metadatas'][0]]
            
            if relevant_docs:
                precision = len(set(retrieved_filenames) & set(relevant_docs)) / len(retrieved_filenames)
                recall = len(set(retrieved_filenames) & set(relevant_docs)) / len(relevant_docs)
                
                mrr = 0
                for i, doc in enumerate(retrieved_filenames, 1):
                    if doc in relevant_docs:
                        mrr = 1.0 / i
                        break
                
                precisions.append(precision)
                recalls.append(recall)
                mrr_scores.append(mrr)
            
            time.sleep(0.2)  # Rate limiting
        
        self.metrics['retrieval'] = {
            'precision@10': np.mean(precisions) if precisions else 0,
            'recall@10': np.mean(recalls) if recalls else 0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0,
            'num_test_queries': len(test_queries)
        }
        
        print(f"\n📊 Retrieval Quality:")
        print(f"   Precision@10: {self.metrics['retrieval']['precision@10']:.3f} ({self.metrics['retrieval']['precision@10']*100:.1f}%)")
        print(f"   Recall@10:    {self.metrics['retrieval']['recall@10']:.3f} ({self.metrics['retrieval']['recall@10']*100:.1f}%)")
        print(f"   MRR:          {self.metrics['retrieval']['mrr']:.3f}")
    
    # GENERATION METRICS
    def calculate_generation_metrics(self, test_queries: List[Dict]):
        print("\n" + "="*70)
        print("   GENERATION ACCURACY METRICS")
        print("="*70)
        
        faithfulness_scores = []
        citation_counts = []
        
        for test_case in test_queries:
            query = test_case['query']
            
            query_embedding = self._generate_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            documents = results['documents'][0]
            answer = self._generate_answer(query, documents)
            
            # Check citations
            import re
            citations = re.findall(r'\[Source \d+\]', answer)
            citation_counts.append(len(citations))
            
            # Simple faithfulness check
            has_citations = len(citations) > 0
            is_reasonable_length = 50 < len(answer.split()) < 500
            faithfulness = 1.0 if (has_citations and is_reasonable_length) else 0.5
            faithfulness_scores.append(faithfulness)
            
            time.sleep(0.5)  # Rate limiting
        
        self.metrics['generation'] = {
            'faithfulness': np.mean(faithfulness_scores),
            'avg_citations': np.mean(citation_counts),
            'num_queries': len(test_queries)
        }
        
        print(f"\n🎯 Generation Quality:")
        print(f"   Faithfulness:   {self.metrics['generation']['faithfulness']:.3f} ({self.metrics['generation']['faithfulness']*100:.1f}%)")
        print(f"   Avg Citations:  {self.metrics['generation']['avg_citations']:.1f} per answer")
    
    # RELEVANCE METRICS
    def calculate_relevance_metrics(self, test_queries: List[Dict]):
        print("\n" + "="*70)
        print("   RELEVANCE & USER SATISFACTION")
        print("="*70)
        
        satisfaction_scores = []
        
        for test_case in test_queries:
            query = test_case['query']
            
            query_embedding = self._generate_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            documents = results['documents'][0]
            distances = results['distances'][0]
            answer = self._generate_answer(query, documents)
            
            # Calculate satisfaction score
            avg_relevance = 1 - np.mean(distances)
            has_citations = '[Source' in answer
            answer_length = len(answer.split())
            
            score = avg_relevance * 0.5
            if has_citations:
                score += 0.3
            if 50 <= answer_length <= 500:
                score += 0.2
            
            satisfaction_scores.append(min(score, 1.0))
            time.sleep(0.5)  # Rate limiting
        
        self.metrics['relevance'] = {
            'avg_relevance': np.mean([1 - d for result in [self.collection.query(query_embeddings=[self._generate_embedding(tq['query'])], n_results=5) for tq in test_queries] for d in result['distances'][0]]),
            'user_satisfaction': np.mean(satisfaction_scores),
            'satisfaction_rate': sum(1 for s in satisfaction_scores if s >= 0.7) / len(satisfaction_scores)
        }
        
        print(f"\n⭐ Relevance & Satisfaction:")
        print(f"   Avg Relevance:      {self.metrics['relevance']['avg_relevance']:.3f} ({self.metrics['relevance']['avg_relevance']*100:.1f}%)")
        print(f"   User Satisfaction:  {self.metrics['relevance']['user_satisfaction']:.3f} ({self.metrics['relevance']['user_satisfaction']*100:.1f}%)")
        print(f"   Satisfaction Rate:  {self.metrics['relevance']['satisfaction_rate']:.3f} ({self.metrics['relevance']['satisfaction_rate']*100:.1f}%)")
    
    # LATENCY METRICS
    def calculate_latency_metrics(self, test_queries: List[Dict]):
        print("\n" + "="*70)
        print("   LATENCY & EFFICIENCY METRICS")
        print("="*70)
        
        embedding_times = []
        retrieval_times = []
        generation_times = []
        total_times = []
        
        for test_case in test_queries:
            query = test_case['query']
            
            total_start = time.time()
            
            # Embedding
            embed_start = time.time()
            query_embedding = self._generate_embedding(query)
            embedding_times.append(time.time() - embed_start)
            
            # Retrieval
            retrieval_start = time.time()
            results = self.collection.query(query_embeddings=[query_embedding], n_results=5)
            retrieval_times.append(time.time() - retrieval_start)
            
            # Generation
            gen_start = time.time()
            documents = results['documents'][0]
            answer = self._generate_answer(query, documents)
            generation_times.append(time.time() - gen_start)
            
            total_times.append(time.time() - total_start)
            time.sleep(0.5)  # Rate limiting
        
        self.metrics['latency'] = {
            'avg_embedding_time': np.mean(embedding_times),
            'avg_retrieval_time': np.mean(retrieval_times),
            'avg_generation_time': np.mean(generation_times),
            'avg_total_time': np.mean(total_times),
            'p95_total_time': np.percentile(total_times, 95),
            'p99_total_time': np.percentile(total_times, 99)
        }
        
        print(f"\n⚡ Latency:")
        print(f"   Avg Embedding:   {self.metrics['latency']['avg_embedding_time']*1000:.0f}ms")
        print(f"   Avg Retrieval:   {self.metrics['latency']['avg_retrieval_time']*1000:.0f}ms")
        print(f"   Avg Generation:  {self.metrics['latency']['avg_generation_time']*1000:.0f}ms")
        print(f"   Avg Total:       {self.metrics['latency']['avg_total_time']*1000:.0f}ms ({self.metrics['latency']['avg_total_time']:.2f}s)")
        print(f"   P95 Total:       {self.metrics['latency']['p95_total_time']*1000:.0f}ms")
    
    def save_metrics(self, filepath: str = "evaluation_results_gemini.json"):
        results = {
            'timestamp': datetime.now().isoformat(),
            'ai_provider': 'Google Gemini',
            'models': {
                'embeddings': 'text-embedding-004',
                'chat': 'gemini-1.5-flash-latest'
            },
            'metrics': self.metrics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Metrics saved to: {filepath}")

def create_test_queries():
    return [
        {
            'query': 'What are telecommunications laws?',
            'relevant_docs': ['telecommunications-act-2053.pdf'],
        },
        {
            'query': 'नागरिकता कानून के हो?',
            'relevant_docs': ['citizenship-act.pdf'],
        },
        {
            'query': 'What are property rights?',
            'relevant_docs': ['property-act.pdf'],
        },
        {
            'query': 'सञ्चार सम्बन्धी कानून',
            'relevant_docs': ['telecommunications-act.pdf'],
        },
        {
            'query': 'media regulation laws',
            'relevant_docs': ['media-regulation.pdf'],
        }
    ]

def main():
    print("="*70)
    print("   RAG SYSTEM EVALUATION (GEMINI)")
    print("="*70)
    
    try:
        evaluator = RAGEvaluator()
    except ValueError as e:
        print(f"❌ {e}")
        print("\n📝 Create a .env file with: GEMINI_API_KEY=your-key-here")
        return
    
    test_queries = create_test_queries()
    
    print(f"\n📝 Running evaluation with {len(test_queries)} test queries...")
    
    evaluator.calculate_retrieval_metrics(test_queries)
    evaluator.calculate_generation_metrics(test_queries)
    evaluator.calculate_relevance_metrics(test_queries)
    evaluator.calculate_latency_metrics(test_queries)
    
    evaluator.save_metrics("evaluation_results_gemini.json")
    
    print("\n" + "="*70)
    print("   EVALUATION SUMMARY")
    print("="*70)
    print(f"   Retrieval Quality:  {evaluator.metrics['retrieval']['precision@10']*100:.1f}%")
    print(f"   Generation Quality: {evaluator.metrics['generation']['faithfulness']*100:.1f}%")
    print(f"   User Satisfaction:  {evaluator.metrics['relevance']['user_satisfaction']*100:.1f}%")
    print(f"   Avg Response Time:  {evaluator.metrics['latency']['avg_total_time']*1000:.0f}ms")
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()