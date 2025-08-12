from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.prompts import PromptTemplate
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={
    r"/chat": {
        "origins": ["https://www.herhaq.org", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    },
    r"/api/chat": {
        "origins": ["https://www.herhaq.org", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Set your system prompt here
SYSTEM_PROMPT = (
    "You are a helpful, motivational sister who answers in a supportive, encouraging tone, mixing simple Urdu and English. Always address the user as 'behn' and keep answers concise, clear, and empathetic.\n\n"
    "Context:\n{context_str}\n\nQuestion: {query_str}\n\nAnswer:"
)

# Initialize LlamaIndex components once
os.environ["COHERE_API_KEY"] = "V8s5d0zNwgRu89WWzwdZifqhBvgM1oqLBr5HcJL1"
embed_model = CohereEmbedding(
    cohere_api_key=os.environ["COHERE_API_KEY"]
)
llm = Cohere(api_key=os.environ["COHERE_API_KEY"])

Settings.embed_model = embed_model
Settings.llm = llm

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Use the system prompt as a custom prompt template
custom_prompt = PromptTemplate(SYSTEM_PROMPT)
query_engine = index.as_query_engine(text_qa_template=custom_prompt)

HTML_TEMPLATE = '''
<!doctype html>
<title>HerHaq</title>
<h1>Ask HerHaq</h1>
<form method=post>
  <input name=query size=60 value="{{ query|default('') }}">
  <input type=submit value=Ask>
</form>
{% if response %}
  <h2>Answer:</h2>
  <div style="white-space: pre-wrap; border:1px solid #ccc; padding:10px;">{{ response }}</div>
{% endif %}
'''

# Custom tone postprocessor
def make_motivational_sister(text):
    intro = "Behn, himmat na haaro! Yeh maloomat aap ke liye hai:"
    outro = "\n\nAap apne haqooq jaanti rahiye, hum aap ke saath hain! 💪"
    motivational_response = f"{intro}\n\n{text}\n{outro}"
    replacements = {
        "rights": "haqooq",
        "women": "khawateen",
        "support": "madad",
        "harassment": "tang karna",
        "help": "madad",
    }
    for eng, urdu in replacements.items():
        motivational_response = motivational_response.replace(eng, f"{eng} ({urdu})")
    return motivational_response

@app.route('/', methods=['GET', 'POST'])
def home():
    response = None
    query = ''
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            try:
                raw_response = query_engine.query(query)
                response = str(raw_response)
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                response = "Sorry behn, kuch masla ho gaya. Please try again."
    return render_template_string(HTML_TEMPLATE, response=response, query=query)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        # Validate data structure
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_query = data.get('query', '')
        
        # Validate query
        if not user_query or not isinstance(user_query, str) or user_query.strip() == '':
            return jsonify({'error': 'Query is required and must be a non-empty string'}), 400
        
        # Process query
        raw_response = query_engine.query(user_query.strip())
        return jsonify({'answer': str(raw_response)})
        
    except Exception as e:
        logger.error(f"Error in API chat: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    try:
        # Handle OPTIONS request for CORS preflight
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', 'https://www.herhaq.org')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
        
        # Handle POST request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_query = data.get('query', '')
        
        # Enhanced validation
        if not user_query or not isinstance(user_query, str) or user_query.strip() == '':
            return jsonify({
                'error': 'Query is required and must be a non-empty string',
                'details': 'Please provide a valid question in the query field'
            }), 400
        
        # Validate query length
        if len(user_query.strip()) < 3:
            return jsonify({
                'error': 'Query too short',
                'details': 'Please provide a more detailed question (minimum 3 characters)'
            }), 400
        
        if len(user_query.strip()) > 500:
            return jsonify({
                'error': 'Query too long',
                'details': 'Please limit your question to 500 characters'
            }), 400
        
        # Process the query
        raw_response = query_engine.query(user_query.strip())
        motivational_response = make_motivational_sister(str(raw_response))
        
        response = jsonify({
            'answer': motivational_response,
            'success': True,
            'query_length': len(user_query.strip())
        })
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', 'https://www.herhaq.org')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': 'Unable to process your request. Please try again later.'
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': str(__import__('datetime').datetime.now())
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting HerHaq server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
