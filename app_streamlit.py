import streamlit as st
import pandas as pd
import os
from groq import Groq
import folium
from streamlit_folium import st_folium
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document



st.set_page_config(page_title="Saruê ::: Fiocruz Brasília", layout="wide")

st.title("Saruê ::: Fiocruz Brasília")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_data(show_spinner=False)
def load_documents():
    base_dir = os.path.join(os.path.dirname(__file__), "samples")
    paths = ["noticias_ses_1_100.csv", "fiocruz_noticias.csv", "min_saude.csv"]

    documents = []

    for path in paths:
        df = pd.read_csv(os.path.join(base_dir, path),sep=";",engine="python",on_bad_lines="skip",encoding="utf-8")
        st.write(df.columns)

        for _, row in df.iterrows():
            titulo = str(row.get("title", "")).strip()
            noticia = str(row.get("content", "")).strip()

            text = f"{titulo}\n\n{noticia}".strip()

            if len(text) > 50:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path}
                    )
                )

    return documents

@st.cache_resource(show_spinner=False)
def setup_retriever():
    docs = load_documents()

    st.write(docs)

    splitter = CharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    st.write("Total de documentos:", len(docs))
    st.write("Total de chunks:", len(chunks))

    db = FAISS.from_documents(chunks, embedding=embeddings)

    return db.as_retriever(search_kwargs={"k": 5})

retriever = setup_retriever()

col_map, col_chat = st.columns([1, 1])

with col_map:
    st.subheader("Mapa da UBS")

    m = folium.Map(location=[-15.793889, -47.882778], zoom_start=12, tiles="CartoDB positron")

    folium.Marker([-15.793889, -47.882778], popup="UBS Central", tooltip="UBS Central").add_to(m)

    st_folium(m, width=500, height=450)

with col_chat:
    st.subheader("Assistente em Saúde Pública (RAG)")

    pergunta = st.text_input("Digite sua pergunta:")

    if pergunta:
        with st.spinner("Buscando evidências nos documentos..."):
            documentos = retriever.invoke(pergunta)

            contexto = "\n\n---\n\n".join(
                [doc.page_content for doc in documentos]
            )

            prompt = f"""
Você é um assistente especializado em saúde pública brasileira.
Responda exclusivamente com base nas informações fornecidas no contexto.
Se a resposta não estiver claramente presente, diga que não encontrou evidência suficiente.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "Você responde de forma técnica, objetiva e em português."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=600
            )

            resposta = response.choices[0].message.content

            st.markdown("Resposta")
            st.write(resposta)


st.markdown("---")
st.markdown(
    "© 2026 – **Saruê** – Fiocruz Brasília  \n"
    "Grupo de Inteligência Computacional na Saúde (GICS)"
)
