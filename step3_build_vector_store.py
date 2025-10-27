"""
STEP 3: Build Vector Store
"""
import json
import chromadb
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class VectorStoreBuilder:
    def __init__(self, db_path="chroma_db"):
        print(f"🗄️  Initializing ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None
        
    def create_collection(self, collection_name="legal_documents", reset=False):
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"   Deleted existing collection")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Nepali legal documents", "created_at": datetime.now().isoformat()}
        )
        print(f"   ✓ Collection ready: {collection_name}")
        
    def sanitize_metadata(self, metadata):
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
        total = len(embedded_chunks)
        print(f"\n📥 Adding {total} documents to vector store...")
        
        for i in tqdm(range(0, total, batch_size), desc="Adding batches"):
            batch = embedded_chunks[i:i + batch_size]
            ids = [f"doc_{i + j}" for j in range(len(batch))]
            documents = [chunk['text'] for chunk in batch]
            embeddings = [chunk['embedding'] for chunk in batch]
            metadatas = [self.sanitize_metadata(chunk['metadata']) for chunk in batch]
            
            self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        
        print(f"   ✓ Successfully added {total} documents")
    
    def get_stats(self):
        count = self.collection.count()
        return {'total_documents': count, 'collection_name': self.collection.name}

def main():
    print("=" * 70)
    print("   STEP 3: BUILDING VECTOR STORE")
    print("=" * 70)
    
    embeddings_file = Path("data/embeddings/embedded_chunks.json")
    if not embeddings_file.exists():
        print("❌ Embeddings file not found!")
        return
    
    print(f"\n📖 Loading embedded chunks...")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embedded_chunks = json.load(f)
    print(f"   Loaded {len(embedded_chunks)} chunks")
    
    builder = VectorStoreBuilder(db_path="chroma_db")
    builder.create_collection(collection_name="legal_documents", reset=True)
    builder.add_documents(embedded_chunks, batch_size=100)
    
    stats = builder.get_stats()
    print(f"\n📊 Vector Store Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    
    print("\n" + "=" * 70)
    print("   VECTOR STORE BUILT SUCCESSFULLY!")
    print("=" * 70)
    print("✓ Database location: chroma_db/")
    print("\n🎯 Next step: Run step4_test_rag.py")

if __name__ == "__main__":
    main()
