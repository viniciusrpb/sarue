from flask import Flask, request, render_template, jsonify
import streamlit as st
import os
from dotenv import load_dotenv
import spacy
import os
import json

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if OPENROUTER_API_KEY is None:
    st.error("OPENROUTER_API_KEY não foi definida!")

MODEL_NAME = "meta-llama/llama-3-70b-instruct"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ubs')
def get_ubss():
    inputData = os.path.join('samples', 'ubs_data.json')
    f = open(inputData, 'r', encoding='utf-8')
    data = json.load(f)
    return jsonify(data)

@app.route('/api/data')
def get_data():
    return jsonify({"message": "Centro de Saúde n 13 - Asa Norte", "coords": [-15.7432347,-47.8915867]})

# pip install flask langchain openai
@app.route('/api/llm-chat', methods=['POST'])
def llm_chat():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Pergunta não fornecida."}), 400

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return jsonify({"error": "Erro ao consultar o LLM."}), 500

    response_json = response.json()
    answer = response_json['choices'][0]['message']['content']

    return jsonify({"answer": answer})

@app.route('/api/ubs-info')
def get_info():
    file_path = os.path.join('samples', 'database.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
