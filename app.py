from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.prompts import PromptTemplate

import os

app = Flask(__name__)
CORS(app, origins=['https://herhaq.org', 'https://www.herhaq.org'], supports_credentials=True)

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
            raw_response = query_engine.query(query)
            response = str(raw_response)
    return render_template_string(HTML_TEMPLATE, response=response, query=query)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    raw_response = query_engine.query(user_query)
    return jsonify({'answer': str(raw_response)})

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle CORS preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    # Handle POST request
    data = request.get_json()
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    
    raw_response = query_engine.query(user_query)
    motivational_response = make_motivational_sister(str(raw_response))
    
    return jsonify({'answer': motivational_response})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))  # Render provides PORT env var
    app.run(host='0.0.0.0', port=port)
