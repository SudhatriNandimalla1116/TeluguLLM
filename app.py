from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)

# HuggingFace API Setup
HF_TOKEN = "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"  # Replace with your actual token
HF_API_URL = "https://api-inference.huggingface.co/models/moonshotai/Kimi-K2-Instruct"


headers = {"Authorization": "Bearer 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"}

@app.route('/')
def serve_demo():
    return send_from_directory('.', 'demo.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data.get('text')

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    payload = {
        "inputs": user_input,
        "parameters": {"max_new_tokens": 100},
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, list) and 'generated_text' in result[0]:
            output_text = result[0]['generated_text']
        else:
            output_text = result.get('generated_text', 'సమాధానం దొరకలేదు.')

        return jsonify({'response': output_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Failed to fetch response from model'}), 500

if __name__ == '__main__':
    app.run(debug=True)
