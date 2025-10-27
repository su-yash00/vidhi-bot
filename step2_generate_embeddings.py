"""
STEP 2: Generate Embeddings (FIXED)
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file!")
        
        # Set the key in environment (new client reads it from env)
        os.environ["OPENAI_API_KEY"] = api_key
        
        self.client = OpenAI()
        self.model = "text-embedding-3-large"
    
    def truncate_text(self, text, max_tokens=8000):
        """Truncate text to fit within token limit"""
        # Rough estimate: 1 token ≈ 0.75 words
        max_words = int(max_tokens * 0.75)
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text
    
    def generate_single_embedding(self, text):
        """Generate embedding for a single text"""
        try:
            # Truncate if needed
            text = self.truncate_text(text)
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error: {e}")
            return [0] * 3072
    
    def generate_batch_embeddings(self, texts):
        """Generate embeddings one by one (safer)"""
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        for i, text in enumerate(tqdm(texts, desc="Processing chunks")):
            try:
                embedding = self.generate_single_embedding(text)
                embeddings.append(embedding)
                
                # Rate limiting
                if i % 10 == 0:
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"\n❌ Failed on chunk {i}: {e}")
                embeddings.append([0] * 3072)
        
        return embeddings
    
    def process_chunks(self, chunks_file, output_file):
        """Process all chunks and generate embeddings"""
        print(f"📖 Loading chunks from: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"   Found {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Calculate cost estimate (more accurate now)
        total_words = sum(len(text.split()) for text in texts)
        total_tokens = int(total_words * 1.3)
        cost_estimate = (total_tokens / 1_000_000) * 0.13
        print(f"\n💰 Estimated cost: ${cost_estimate:.2f}")
        print(f"   Total tokens (approx): {total_tokens:,}")
        
        # Ask for confirmation
        response = input("\nProceed with embedding generation? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled")
            return
        
        # Generate embeddings
        print(f"\n🔄 Generating embeddings...")
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
        print(f"✓ Actual cost: ~${cost_estimate:.2f}")


def main():
    print("=" * 70)
    print("   STEP 2: GENERATING EMBEDDINGS (FIXED VERSION)")
    print("=" * 70)
    
    chunks_file = Path("data/processed/all_chunks.json")
    if not chunks_file.exists():
        print("❌ Chunks file not found!")
        return
    
    try:
        generator = EmbeddingGenerator()
    except ValueError as e:
        print(e)
        print("\n📝 Create a .env file with: OPENAI_API_KEY=sk-your-key")
        return
    
    output_file = Path("data/embeddings/embedded_chunks.json")
    generator.process_chunks(chunks_file, output_file)
    
    print("\n" + "=" * 70)
    print("   EMBEDDING GENERATION COMPLETE!")
    print("=" * 70)
    print("\n🎯 Next step: Run step3_build_vector_store.py")


if __name__ == "__main__":
    main()