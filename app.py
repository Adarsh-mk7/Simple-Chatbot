from flask import Flask, request, render_template
from flask_cors import CORS
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Initialize the modern Causal Instruction Model
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
print(f"Loading smart instruction model: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.unk_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)

# Initialize system prompt to dictate the AI's behavior
messages = [
    {
        "role": "system",
        "content": "You are a helpful, logical AI assistant. Answer the user's questions accurately, including math and reasoning tasks."
    }
]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    global messages
    
    # 1. Parse incoming user message
    data = json.loads(request.get_data(as_text=True))
    user_input = data['prompt']

    # 2. Append to structured message context
    messages.append({"role": "user", "content": user_input})
    
    # 3. Maintain sliding context window (Keep system prompt + last 10 turns)
    messages = [messages[0]] + messages[-10:]

    # 4. Apply the structural Chat Template required by causal LLMs
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        max_length=512
    )

    # 5. Generate precise tokens using Causal Inference parameters
    with torch.inference_mode():
        outputs = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=60,
            temperature=0.3,          # Lower temperature = more precise factual/math tracking
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.3,   # Prevents infinite repetition loops
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id
        )

    # 6. Slice out the prompt tokens so it only decodes the new answer
    response = tokenizer.decode(
        outputs[0][tokenized["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # 7. Append assistant response back to structural tracking context
    messages.append({"role": "assistant", "content": response})

    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)