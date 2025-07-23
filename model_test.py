"""
Standalone script to test the LLM locally
Tests basic model functionality before Flask integration
"""
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

def setup_model():
    """
    Initialize the local LLM model
    Using TinyLlama for fast CPU inference
    """
    try:
        # Download model using huggingface_hub
        os.makedirs("models", exist_ok=True)
        
        # Download model
        print("Downloading TinyLlama model...")
        model_path = hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        
        print(f"Model downloaded to: {model_path}")
        
        # Load model
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def inject_business_knowledge(prompt):
    """
    Inject business knowledge about TorkeHub
    """
    context = """
    Business Knowledge:
    - TorkeHub is a white-label CRM platform for enterprise clients
    - TorkeHub specializes in customer relationship management solutions
    - TorkeHub offers customizable dashboards and analytics tools
    
    User Question: """
    
    return context + prompt

def get_response(model, prompt):
    """
    Generate response from the model
    """
    try:
        # Inject business knowledge
        enhanced_prompt = inject_business_knowledge(prompt)
        
        # Generate response
        response = model(
            enhanced_prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["User:", "Question:"]
        )
        
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    """
    Main testing function
    """
    print("🤖 Initializing TinyLlama model...")
    model = setup_model()
    
    if not model:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    print("💡 Type 'quit' to exit\n")
    
    # Test with sample prompts
    test_prompts = [
        "What is TorkeHub?",
        "Tell me about TorkeHub's services",
        "How can TorkeHub help businesses?"
    ]
    
    print("🧪 Testing with sample prompts:")
    for prompt in test_prompts:
        print(f"\n🔹 Prompt: {prompt}")
        response = get_response(model, prompt)
        print(f"🤖 Response: {response}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("🗣️  Interactive Mode - Ask anything!")
    print("="*50)
    
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        if user_input:
            response = get_response(model, user_input)
            print(f"🤖 Bot: {response}")

if __name__ == "__main__":
    main()