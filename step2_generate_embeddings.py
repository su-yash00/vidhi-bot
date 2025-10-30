"""
STEP 3: Generate Embeddings for Legal Chunks
This script reads all processed PDF chunks and generates embeddings using OpenAI's embedding model.
Embeddings are saved alongside the chunk metadata for later use in the vector store (ChromaDB).
"""

import json
import time
from pathlib import Path
import os
from openai import OpenAI
from tqdm import tqdm
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make sure the API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment!")

# Set API key in environment for OpenAI client
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)  # Initialize OpenAI client

class EmbeddingGenerator:
    """
    Handles embedding generation for text chunks.
    Supports truncation to respect token limits and batch processing.
    """
    def __init__(self, model="text-embedding-3-large", max_tokens=8000):
        self.model = model
        self.max_tokens = max_tokens
        # Initialize tokenizer for accurate token counting
        self.enc = tiktoken.get_encoding("cl100k_base")

    def truncate_text(self, text):
        """
        Truncate text to fit within the max_tokens limit.
        This prevents errors from sending too large inputs to the API.
        """
        tokens = self.enc.encode(text)
        if len(tokens) > self.max_tokens:
            return self.enc.decode(tokens[:self.max_tokens])
        return text

    def generate_embedding(self, text):
        """
        Generate an embedding for a single chunk of text.
        Returns a fallback vector of zeros if the API fails.
        """
        try:
            text = self.truncate_text(text)
            response = client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0] * 3072  # fallback vector

    def generate_batch_embeddings(self, texts, sleep_time=0.1):
        """
        Generate embeddings for a list of texts, one by one.
        Adds a small delay to avoid rate limits.
        """
        embeddings = []
        for i, text in enumerate(tqdm(texts, desc="Generating embeddings")):
            embeddings.append(self.generate_embedding(text))
            # Add a slightly longer delay every 10 chunks
            if i % 10 == 0:
                time.sleep(sleep_time * 5)
            else:
                time.sleep(sleep_time)
        return embeddings

def main():
    """
    Main function to:
    1. Load all text chunks from JSON.
    2. Generate embeddings.
    3. Save embedded chunks with metadata (including PDF info).
    """
    chunks_file = Path("data/processed/all_chunks_cleaned.json")
    if not chunks_file.exists():
        print("Chunks file not found! Please run Step 2 first.")
        return

    # Load all processed chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Remove any empty chunks to prevent errors
    chunks = [c for c in chunks if c.get("text")]
    print(f" {len(chunks)} valid chunks found for embedding generation")

    # Initialize the embedding generator
    generator = EmbeddingGenerator()

    # Extract the text from each chunk
    texts = [c['text'] for c in chunks]

    # Estimate total token count and cost
    total_tokens = sum(len(generator.enc.encode(t)) for t in texts)
    estimated_cost = (total_tokens / 1_000_000) * 0.13
    print(f" Estimated token count: {total_tokens:,}, approximate cost: ${estimated_cost:.2f}")

    # Ask for user confirmation before generating embeddings
    proceed = input("Proceed with embedding generation? (yes/no): ")
    if proceed.lower() not in ["yes", "y"]:
        print("Cancelled by user.")
        return

    # Generate embeddings in a safe batch manner
    embeddings = generator.generate_batch_embeddings(texts)

    # Combine each chunk with its embedding and metadata
    embedded_chunks = []
    for chunk, emb in zip(chunks, embeddings):
        # Ensure each chunk has a 'metadata' dictionary with PDF info
        metadata = chunk.get('metadata', {})
        metadata.setdefault('filename', chunk.get('filename', 'Unknown.pdf'))
        metadata.setdefault('folder', chunk.get('folder', 'data/PDFs'))
        embedded_chunks.append({
            "text": chunk['text'],
            "embedding": emb,
            "metadata": metadata
        })

    # Save the embedded chunks to JSON
    output_file = Path("data/embeddings/embedded_chunks.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(embedded_chunks)} embedded chunks to {output_file}")
    print("Next: Run Step 4 to build the vector store using these embeddings.")

if __name__ == "__main__":
    main()
