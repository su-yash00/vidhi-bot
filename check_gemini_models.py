"""
Check available Gemini models - SIMPLIFIED VERSION
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def check_models():
    """List all available Gemini models"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file!")
        return
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to configure API: {e}")
        return
    
    print("=" * 70)
    print("   AVAILABLE GEMINI MODELS")
    print("=" * 70)
    
    print("\n📋 Listing all models...\n")
    
    # Get all models
    embedding_models = []
    chat_models = []
    
    for model in genai.list_models():
        model_name = model.name
        supported_methods = [m for m in model.supported_generation_methods]
        
        # Check if it's an embedding model
        if 'embedContent' in supported_methods:
            embedding_models.append(model_name)
        
        # Check if it's a chat/generation model
        if 'generateContent' in supported_methods:
            chat_models.append(model_name)
    
    # Display embedding models
    print("🔢 EMBEDDING MODELS:")
    print("-" * 70)
    if embedding_models:
        for model in embedding_models:
            print(f"   ✓ {model}")
    else:
        print("   No embedding models found")
    
    # Display chat models
    print("\n💬 CHAT/GENERATION MODELS:")
    print("-" * 70)
    if chat_models:
        for model in chat_models:
            print(f"   ✓ {model}")
    else:
        print("   No chat models found")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("   RECOMMENDED MODELS")
    print("=" * 70)
    
    print("\n🔢 For Embeddings:")
    print("   models/text-embedding-004")
    
    print("\n💬 For Chat:")
    print("   models/gemini-1.5-flash-latest  (Fast, efficient)")
    print("   models/gemini-1.5-pro-latest    (More capable)")
    print("   models/gemini-2.0-flash-exp     (Experimental, newest)")
    
    print("\n" + "=" * 70)
    print("\n✅ Use these model names in your code!")

def test_models():
    """Test embedding and chat models"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        return
    
    genai.configure(api_key=api_key)
    
    print("\n" + "=" * 70)
    print("   TESTING MODELS")
    print("=" * 70)
    
    # Test embedding
    print("\n🔢 Testing Embedding Model...")
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content="test",
            task_type="retrieval_document"
        )
        print(f"   ✓ Success! Dimensions: {len(result['embedding'])}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test chat models
    chat_models = [
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro-latest",
        "models/gemini-2.0-flash-exp"
    ]
    
    print("\n💬 Testing Chat Models...")
    for model_name in chat_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                "Say 'test'",
                generation_config=genai.types.GenerationConfig(max_output_tokens=10)
            )
            print(f"   ✓ {model_name} - Working!")
        except Exception as e:
            print(f"   ❌ {model_name} - Failed: {str(e)[:50]}")

if __name__ == "__main__":
    try:
        check_models()
        test_models()
        
        print("\n" + "=" * 70)
        print("   RECOMMENDED CONFIGURATION")
        print("=" * 70)
        print("\nUse in your code:")
        print("   embedding_model = 'models/text-embedding-004'")
        print("   chat_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your GEMINI_API_KEY in .env")
        print("2. Make sure you have internet connection")
        print("3. Verify your API key at: https://makersuite.google.com/app/apikey")