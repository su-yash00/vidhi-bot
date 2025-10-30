"""
STEP 4: Build Vector Store
This script takes all embedded chunks and stores them in ChromaDB.
The vector store allows for fast similarity search for your RAG system.
"""

import json
from pathlib import Path
from datetime import datetime
import chromadb
from tqdm import tqdm

class VectorStoreBuilder:
    """
    Handles creation and population of a Chroma vector store.
    """
    def __init__(self, db_path="chroma_db"):
        print(f"  Initializing ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None

    def create_collection(self, collection_name="legal_documents", reset=False):
        """
        Create a collection in ChromaDB.
        If reset=True, any existing collection with the same name will be deleted.
        """
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"   Deleted existing collection: {collection_name}")
            except Exception:
                pass

        # Create or retrieve the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Nepali legal documents",
                "created_at": datetime.now().isoformat()
            }
        )
        print(f"    Collection ready: {collection_name}")

    @staticmethod
    def sanitize_metadata(metadata):
        """
        Ensure that all metadata fields are serializable for ChromaDB.
        Convert non-primitive types to strings.
        """
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        return clean

    def add_documents(self, embedded_chunks, batch_size=100):
        """
        Add embedded chunks to the Chroma collection in batches.
        """
        total = len(embedded_chunks)
        print(f"\nAdding {total} documents to vector store...")

        for i in tqdm(range(0, total, batch_size), desc="Adding batches"):
            batch = embedded_chunks[i:i + batch_size]

            # Assign unique IDs for each chunk
            ids = [f"doc_{i + j}" for j in range(len(batch))]
            documents = [chunk['text'] for chunk in batch]
            embeddings = [chunk['embedding'] for chunk in batch]
            metadatas = [self.sanitize_metadata(chunk['metadata']) for chunk in batch]

            # Add batch to Chroma
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

        print(f"  Successfully added {total} documents")

    def get_stats(self):
        """
        Retrieve simple stats about the collection.
        """
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }


def main():
    """
    Main function to load embeddings and build the Chroma vector store.
    """
    print("=" * 70)
    print("   STEP 4: BUILDING VECTOR STORE")
    print("=" * 70)

    embeddings_file = Path("data/embeddings/embedded_chunks.json")
    if not embeddings_file.exists():
        print(" Embeddings file not found! Please run Step 3 first.")
        return

    # Load all embedded chunks
    print("\nLoading embedded chunks...")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embedded_chunks = json.load(f)
    print(f"   Loaded {len(embedded_chunks)} chunks")

    # Initialize vector store builder
    builder = VectorStoreBuilder(db_path="chroma_db")
    builder.create_collection(collection_name="legal_documents", reset=True)

    # Add all documents in batches
    builder.add_documents(embedded_chunks, batch_size=100)

    # Print summary stats
    stats = builder.get_stats()
    print(f"\nVector Store Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print("=" * 70)
    print(" Vector store built successfully!")
    print("Database location: chroma_db/")
    print("\nNext step: Run Step 5 to test the RAG system with this vector store.")


if __name__ == "__main__":
    main()
