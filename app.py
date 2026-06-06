from flask import Flask

app = Flask(__name__)

from flask import Flask, request
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)  # This prevents Cross-Origin Blocking errors from your web browser

# Initialize the model and tokenizer 
model_name = "facebook/blenderbot-400M-distill"
print("Loading BlenderBot into Flask Backend...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    global conversation_history
    
    # Read and parse JSON string from the incoming HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    # Keep only the last 6 exchanges to prevent context length overflow crashes
    conversation_history = conversation_history[-6:]
    history = "\n".join(conversation_history)

    # Tokenize input using modern safe keyword arguments
    inputs = tokenizer(
        text=history,
        text_pair=input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3
    )

    # Decode tokens back to readable plaintext
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Append to state tracking history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    # Run server locally on default port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)