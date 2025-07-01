import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = "meta-llama/llama-3-70b-instruct"

st.set_page_config(page_title="Assistente Saruê", layout="wide")

st.title("Assistente LLM - Saruê")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY não foi definida no ambiente ou .env.")
    st.stop()

question = st.text_area("Digite sua pergunta para o LLM:")

if st.button("Enviar pergunta"):
    if not question.strip():
        st.warning("Digite uma pergunta antes de enviar.")
    else:
        with st.spinner("Consultando o LLM..."):
            try:
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

                if response.status_code == 200:
                    data = response.json()
                    answer = data['choices'][0]['message']['content']
                    st.success("Resposta:")
                    st.markdown(answer)
                else:
                    st.error(f"Erro na resposta: {response.status_code}")
                    st.json(response.json())

            except Exception as e:
                st.error(f"Erro durante a requisição: {e}")
