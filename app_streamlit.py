import streamlit as st
from streamlit_folium import st_folium
import folium
import openai
import os

openai.api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
model_id = "llama-3.3-70b-versatile"

st.set_page_config(layout="wide", page_title="Saruê - Fiocruz Brasília")

col1, col2 = st.columns([1, 1])
with col1:
    st.image("static/imgs/saruevislogo.png", width=160)
with col2:
    st.image("static/imgs/fiocruz.png", width=160)

st.markdown("---")

col_esq, col_dir = st.columns([1, 3])

with col_esq:
    st.subheader("Unidades Básicas de Saúde")

    ubs_list = ["Selecione uma UBS", "UBS 1", "UBS 2", "UBS 3"]
    selected_ubs = st.selectbox("Escolha uma UBS:", ubs_list)

    st.markdown("**Tipo de informação na Tooltip:**")
    tipo_info = st.radio("Escolha:", ["Notícias", "Publicações Oficiais"])

    if tipo_info == "Publicações Oficiais":
        subtipo_licitacoes = st.checkbox("Licitações")
        subtipo_contratos = st.checkbox("Contratos")
        subtipo_pessoal = st.checkbox("Pessoal")

    if st.button("Enviar"):
        st.success(f"Filtrando para: {selected_ubs} - {tipo_info}")

with col_dir:
    mapa_col, bot_col = st.columns([1, 1])

    with mapa_col:
        st.markdown("### Mapa da UBS")

        m = folium.Map(location=[-15.793889, -47.882778], zoom_start=12)

        folium.Marker(
            [-15.793889, -47.882778],
            popup="UBS Central",
            tooltip="Clique aqui"
        ).add_to(m)

        st_folium(m, width=500, height=420)

    with bot_col:
        st.markdown("### Assistente Virtual com LLaMA 3")

        pergunta = st.text_input("Digite sua pergunta:")

        if st.button("Perguntar") and pergunta.strip():
            with st.spinner("Consultando o LLaMA 3 via Groq..."):
                try:
                    resposta = openai.ChatCompletion.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil e preciso, com foco em saúde pública e serviços públicos brasileiros."},
                            {"role": "user", "content": pergunta}
                        ],
                        temperature=0.3
                    )
                    st.markdown("**Resposta:**")
                    st.write(resposta["choices"][0]["message"]["content"])
                except Exception as e:
                    st.error(f"Ocorreu um erro ao consultar o modelo: {e}")

st.markdown("---")
st.markdown("© 2025 - Saruê - Fiocruz Brasília  \nGrupo de Inteligência Computacional na Saúde (GICS)")
