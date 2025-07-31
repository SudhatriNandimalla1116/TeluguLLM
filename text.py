from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

HF_TOKEN = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/moonshotai/Kimi-K2-Instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
payload = {"inputs": "Hello"}

response = requests.post(HF_API_URL, headers=headers, json=payload)
print(response.status_code)
print(response.text)

def generate_kimi_response(user_input: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": user_input,
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Debugging: Print the full response from Hugging Face
        print(f"Hugging Face API Response: {data}")

        if isinstance(data, dict) and data.get("error"):
            return f"Error from model: {data['error']}"
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        
        # If the response format isn't as expected, return raw data or a generic message
        return f"Unexpected response format from model: {data}"
    except requests.exceptions.Timeout:
        print("Error: Hugging Face API request timed out.")
        return "Sorry, the AI took too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"Error during remote LLM generation (RequestException): {e}")
        return "Sorry, there was a network error connecting to the AI."
    except Exception as e:
        print(f"An unexpected error occurred during LLM generation: {e}")
        return "Sorry, an unexpected error occurred."

@app.route('/generate', methods=['POST'])
def generate_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get('text')

    if not user_input:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    llm_response = generate_kimi_response(user_input)
    return jsonify({"response": llm_response})

# Assuming your frontend is set up as in the previous examples
# Choose ONE of these for serving index.html, based on your project structure.
# If index.html is in a 'frontend' subfolder:
@app.route('/index')
def serve_demo():
    return send_from_directory('demo', 'index.html')

# If index.html is in the same directory as app.py:
@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    # Keep debug=True during development for detailed error messages
    app.run(host='0.0.0.0', port=5000, debug=True)
