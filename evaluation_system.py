"""
STEP 6: RAG Evaluation System
Evaluates Retrieval Quality, Generation Accuracy, and User Satisfaction
"""
import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

load_dotenv()

class RAGEvaluator:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found!")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.chroma_client.get_collection(name="legal_documents")
        print("✓ Evaluator initialized")
    
    def evaluate_retrieval_precision(self, query: str, retrieved_docs: List[str], 
                                     retrieved_metadata: List[dict], k: int = 5) -> Dict:
        """
        Evaluate retrieval precision using LLM as judge
        """
        # Take top k results
        docs_to_eval = retrieved_docs[:k]
        metas_to_eval = retrieved_metadata[:k]
        
        relevance_scores = []
        
        for i, (doc, meta) in enumerate(zip(docs_to_eval, metas_to_eval)):
            prompt = f"""Evaluate if this document excerpt is relevant to the query.

Query: {query}

Document excerpt:
{doc[:500]}...

Is this document relevant to answering the query?
Respond with a score from 0-10:
- 0-3: Not relevant
- 4-6: Somewhat relevant
- 7-8: Relevant
- 9-10: Highly relevant

Provide ONLY the numeric score."""

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of document relevance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                score = float(response.choices[0].message.content.strip())
                relevance_scores.append(score / 10.0)  # Normalize to 0-1
            except:
                relevance_scores.append(0.5)  # Default if evaluation fails
        
        precision_at_k = sum(1 for s in relevance_scores if s >= 0.7) / k
        avg_relevance = np.mean(relevance_scores)
        
        return {
            'precision@k': precision_at_k,
            'average_relevance': avg_relevance,
            'relevance_scores': relevance_scores,
            'k': k
        }
    
    def evaluate_generation_faithfulness(self, query: str, generated_answer: str, 
                                        source_documents: List[str]) -> Dict:
        """
        Evaluate if generated answer is faithful to source documents
        """
        context = "\n\n".join(source_documents[:5])
        
        prompt = f"""Evaluate if the generated answer is faithful to the source documents.

Source Documents:
{context}

Generated Answer:
{generated_answer}

Evaluation criteria:
1. Factual consistency: Does the answer contain only information from sources?
2. No hallucinations: Does the answer avoid making up information?
3. Proper representation: Does it accurately represent the source content?

Provide scores (0-10) for:
- Factual Consistency:
- No Hallucinations:
- Proper Representation:
- Overall Faithfulness:

Format: Score only, one per line."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating answer faithfulness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            lines = response.choices[0].message.content.strip().split('\n')
            scores = []
            for line in lines:
                try:
                    score = float(''.join(filter(str.isdigit, line)))
                    if 0 <= score <= 10:
                        scores.append(score / 10.0)
                except:
                    pass
            
            if len(scores) >= 4:
                return {
                    'factual_consistency': scores[0],
                    'no_hallucinations': scores[1],
                    'proper_representation': scores[2],
                    'overall_faithfulness': scores[3]
                }
            else:
                return {
                    'factual_consistency': np.mean(scores) if scores else 0.5,
                    'no_hallucinations': np.mean(scores) if scores else 0.5,
                    'proper_representation': np.mean(scores) if scores else 0.5,
                    'overall_faithfulness': np.mean(scores) if scores else 0.5
                }
        except:
            return {
                'factual_consistency': 0.5,
                'no_hallucinations': 0.5,
                'proper_representation': 0.5,
                'overall_faithfulness': 0.5
            }
    
    def evaluate_answer_quality(self, query: str, generated_answer: str) -> Dict:
        """
        Evaluate overall answer quality and helpfulness
        """
        prompt = f"""Evaluate the quality of this answer to the legal query.

Query: {query}

Answer:
{generated_answer}

Evaluate on these dimensions (0-10):
1. Completeness: Does it fully answer the question?
2. Clarity: Is it clear and easy to understand?
3. Relevance: Does it stay relevant to the question?
4. Usefulness: Would this be helpful to a user?
5. Citation Quality: Are sources properly cited?

Provide scores, one per line."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating answer quality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            lines = response.choices[0].message.content.strip().split('\n')
            scores = []
            for line in lines:
                try:
                    score = float(''.join(filter(str.isdigit, line)))
                    if 0 <= score <= 10:
                        scores.append(score / 10.0)
                except:
                    pass
            
            if len(scores) >= 5:
                return {
                    'completeness': scores[0],
                    'clarity': scores[1],
                    'relevance': scores[2],
                    'usefulness': scores[3],
                    'citation_quality': scores[4],
                    'overall_quality': np.mean(scores)
                }
            else:
                avg = np.mean(scores) if scores else 0.5
                return {
                    'completeness': avg,
                    'clarity': avg,
                    'relevance': avg,
                    'usefulness': avg,
                    'citation_quality': avg,
                    'overall_quality': avg
                }
        except:
            return {
                'completeness': 0.5,
                'clarity': 0.5,
                'relevance': 0.5,
                'usefulness': 0.5,
                'citation_quality': 0.5,
                'overall_quality': 0.5
            }
    
    def evaluate_citation_quality(self, generated_answer: str, 
                                  source_metadata: List[dict]) -> Dict:
        """
        Evaluate quality of citations in the answer
        """
        # Count citations
        import re
        citations = re.findall(r'\[(\d+)\]', generated_answer)
        num_citations = len(citations)
        unique_citations = len(set(citations))
        
        # Check if citations are valid (within range)
        max_valid_citation = len(source_metadata)
        valid_citations = sum(1 for c in citations if int(c) <= max_valid_citation)
        
        # Calculate metrics
        citation_coverage = unique_citations / max_valid_citation if max_valid_citation > 0 else 0
        citation_validity = valid_citations / num_citations if num_citations > 0 else 0
        
        return {
            'total_citations': num_citations,
            'unique_citations': unique_citations,
            'citation_coverage': citation_coverage,
            'citation_validity': citation_validity,
            'citations_per_100_words': num_citations / (len(generated_answer.split()) / 100) if generated_answer else 0
        }
    
    def comprehensive_evaluation(self, query: str, retrieved_docs: List[str],
                                retrieved_metadata: List[dict], distances: List[float],
                                generated_answer: str) -> Dict:
        """
        Perform comprehensive evaluation of the RAG system
        """
        print(f"\n🔍 Evaluating query: {query[:50]}...")
        
        # 1. Retrieval Quality
        print("   📊 Evaluating retrieval quality...")
        retrieval_metrics = self.evaluate_retrieval_precision(query, retrieved_docs, 
                                                              retrieved_metadata, k=5)
        
        # 2. Generation Faithfulness
        print("   🎯 Evaluating generation faithfulness...")
        faithfulness_metrics = self.evaluate_generation_faithfulness(query, generated_answer, 
                                                                     retrieved_docs)
        
        # 3. Answer Quality
        print("   ⭐ Evaluating answer quality...")
        quality_metrics = self.evaluate_answer_quality(query, generated_answer)
        
        # 4. Citation Quality
        print("   📝 Evaluating citation quality...")
        citation_metrics = self.evaluate_citation_quality(generated_answer, retrieved_metadata)
        
        # Calculate overall score
        overall_score = np.mean([
            retrieval_metrics['average_relevance'],
            faithfulness_metrics['overall_faithfulness'],
            quality_metrics['overall_quality']
        ])
        
        return {
            'query': query,
            'retrieval_quality': retrieval_metrics,
            'generation_faithfulness': faithfulness_metrics,
            'answer_quality': quality_metrics,
            'citation_quality': citation_metrics,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_evaluate(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate multiple test cases and aggregate results
        """
        all_results = []
        
        print(f"\n{'='*70}")
        print(f"   BATCH EVALUATION: {len(test_cases)} test cases")
        print(f"{'='*70}")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Processing: {test_case['query'][:50]}...")
            
            result = self.comprehensive_evaluation(
                test_case['query'],
                test_case['retrieved_docs'],
                test_case['retrieved_metadata'],
                test_case['distances'],
                test_case['generated_answer']
            )
            all_results.append(result)
        
        # Aggregate metrics
        aggregate = {
            'total_test_cases': len(test_cases),
            'average_retrieval_precision': np.mean([r['retrieval_quality']['precision@k'] for r in all_results]),
            'average_relevance': np.mean([r['retrieval_quality']['average_relevance'] for r in all_results]),
            'average_faithfulness': np.mean([r['generation_faithfulness']['overall_faithfulness'] for r in all_results]),
            'average_answer_quality': np.mean([r['answer_quality']['overall_quality'] for r in all_results]),
            'average_citation_validity': np.mean([r['citation_quality']['citation_validity'] for r in all_results]),
            'overall_system_score': np.mean([r['overall_score'] for r in all_results]),
            'individual_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return aggregate


def generate_test_cases_from_rag(n_cases: int = 5) -> List[Dict]:
    """
    Generate test cases by querying the RAG system
    """
    from step4_test_rag_enhanced import EnhancedRAG
    
    rag = EnhancedRAG()
    
    test_queries = [
        "What are the main provisions about telecommunications licensing in Nepal?",
        "सञ्चार सम्बन्धी मुख्य कानूनहरू के हुन्?",
        "What are the penalties for violating telecom regulations?",
        "What rights do citizens have regarding privacy?",
        "नागरिकको मौलिक अधिकार के के हुन्?",
    ][:n_cases]
    
    test_cases = []
    
    for query in test_queries:
        print(f"Generating test case for: {query[:50]}...")
        
        # Get RAG response
        results = rag.search_with_reformulation(query, 3)
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        answer = rag.generate_answer_with_citations(query, documents, metadatas, distances)
        
        test_cases.append({
            'query': query,
            'retrieved_docs': documents,
            'retrieved_metadata': metadatas,
            'distances': distances,
            'generated_answer': answer
        })
    
    return test_cases


def main():
    print("=" * 70)
    print("   STEP 6: RAG EVALUATION SYSTEM")
    print("=" * 70)
    
    try:
        evaluator = RAGEvaluator()
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Generate test cases
    print("\n📋 Generating test cases from RAG system...")
    test_cases = generate_test_cases_from_rag(n_cases=3)
    
    # Run evaluation
    results = evaluator.batch_evaluate(test_cases)
    
    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("   EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n📊 Overall Metrics:")
    print(f"   • Retrieval Precision@5: {results['average_retrieval_precision']:.2%}")
    print(f"   • Average Relevance: {results['average_relevance']:.2%}")
    print(f"   • Generation Faithfulness: {results['average_faithfulness']:.2%}")
    print(f"   • Answer Quality: {results['average_answer_quality']:.2%}")
    print(f"   • Citation Validity: {results['average_citation_validity']:.2%}")
    print(f"   • Overall System Score: {results['overall_system_score']:.2%}")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Grade system
    score = results['overall_system_score']
    if score >= 0.9:
        grade = "A+ (Excellent)"
    elif score >= 0.8:
        grade = "A (Very Good)"
    elif score >= 0.7:
        grade = "B (Good)"
    elif score >= 0.6:
        grade = "C (Satisfactory)"
    else:
        grade = "D (Needs Improvement)"
    
    print(f"\n🎯 System Grade: {grade}")
    print("\n" + "=" * 70)
    print("   ✅ EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()