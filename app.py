"""
Flask backend for the AI chatbot
Provides REST API endpoint for chat functionality
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from llama_cpp import Llama
import os
import logging
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global model variable
model = None

def initialize_model(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    models_dir="models",
    n_ctx=2048,
    n_threads=4,
    verbose=False
):
    global model
    try:
        logger.info("🤖 Loading TinyLlama model...")
        """
        Load TinyLlama model. Downloads if not exists locally.
    
        Args:
            repo_id: HuggingFace repository ID
            filename: Model filename
            models_dir: Local directory to store models
            n_ctx: Context window size
            n_threads: Number of CPU threads
            verbose: Enable verbose output
        
        Returns:
            Llama model object or None if failed
        """
    
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, filename)

        if os.path.exists(model_path):
            print(f"✅ Model found locally at: {model_path}")
            print(f"📁 File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        else:
            print(f"📥 Model not found locally. Downloading from {repo_id}...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
        # Load the model here
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose
        )
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def inject_business_context(question):
    """
    Inject TorkeHub business knowledge
    """
    context = """
You are an AI assistant with knowledge about TorkeHub.

Business Information:
- TorkeHub is a white-label CRM platform designed for enterprise clients
- TorkeHub provides customizable customer relationship management solutions
- TorkeHub offers advanced analytics, dashboard customization, and client management tools
- TorkeHub helps businesses streamline their customer interactions and improve sales processes

Please answer the following question based on this context and your general knowledge:

Question: {question}

Answer:"""
    return context.format(question=question)

@app.route('/')
def home():
    """
    Serve the simple frontend (optional)
    """
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Main API endpoint for chat functionality
    Accepts: {"question": "..."}
    Returns: {"answer": "..."}
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "error": "Question field is required and cannot be empty"
            }), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "AI model is not loaded. Please restart the server."
            }), 500
        
        # Generate response
        logger.info(f"📝 Processing question: {question}")

        # Dynamically inject context only for TorkeHub/CRM questions
        keywords = ["torkehub", "crm", "customer relationship", "dashboard", "client management"]
        if any(word in question.lower() for word in keywords):
            enhanced_prompt = inject_business_context(question)
        else:
            # Add "Answer:" to general questions too
            enhanced_prompt = f"Question: {question}\n\nAnswer:"

        response = model(
            enhanced_prompt,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["Question:", "User:", "\n\n"]
        )
        
        answer = response['choices'][0]['text'].strip()
        
        logger.info(f"✅ Generated response successfully")
        
        return jsonify({
            "answer": answer,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status
    })

if __name__ == '__main__':
    # Initialize model before starting server
    if initialize_model():
        print("🚀 Starting Flask server...")
        print("📍 API Endpoint: POST http://localhost:5000/ask")
        print("🌐 Frontend: http://localhost:5000/")
        print("💡 Health Check: GET http://localhost:5000/health")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize model. Server not started.")
