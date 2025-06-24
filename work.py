from flask import Flask, request, render_template, jsonify
import spacy
import os
import json
#from langchain.llms import ...

app = Flask(__name__)

#llm =

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

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    if not question.strip():
        return jsonify({'answer': 'Pergunta vazia.'})
    try:
        response = llm.predict(question)
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'answer': f'Erro: {str(e)}'})

@app.route('/api/ubs-info')
def get_info():
    file_path = os.path.join('samples', 'database.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
