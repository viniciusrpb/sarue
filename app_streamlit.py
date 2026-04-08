import streamlit as st
import pandas as pd
import os
import json
import re
from groq import Groq
import folium
from streamlit_folium import st_folium
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import unicodedata


st.set_page_config(page_title="Saruê ::: Fiocruz Brasília", layout="wide")
st.title("Saruê ::: Fiocruz Brasília")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_data(show_spinner=False)
def load_geojson():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "database", "setoresDF.json")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)

    by_code   = {}
    by_subdist = {}

    def norm(s):
        import unicodedata
        s = unicodedata.normalize("NFKD", s or "")
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

    for feat in gj["features"]:
        props = feat["properties"]
        code  = props.get("CD_SETOR", "")
        sd    = norm(props.get("NM_SUBDIST") or "")

        by_code[code] = feat
        by_subdist.setdefault(sd, []).append(feat)

    return gj, by_code, by_subdist

@st.cache_data(show_spinner=False)
def get_subdist_list():
    _, _, by_subdist = load_geojson()
    return sorted(by_subdist.keys())

@st.cache_data(show_spinner=False)
def load_documents():
    base_dir = os.path.join(os.path.dirname(__file__), "samples")
    paths = ["noticias_ses_1_100.csv"]

    documents = []
    for path in paths:
        df = pd.read_csv(
            os.path.join(base_dir, path),
            sep=",", engine="python",
            on_bad_lines="skip", encoding="utf-8"
        )
        for _, row in df.iterrows():
            titulo  = str(row.get("title",   "")).strip()
            noticia = str(row.get("content", "")).strip()
            text    = f"{titulo}\n\n{noticia}".strip()
            if len(text) > 50:
                documents.append(
                    Document(page_content=text, metadata={"source": path})
                )
    return documents

@st.cache_resource(show_spinner=False)
def setup_retriever():
    docs     = load_documents()
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks   = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = setup_retriever()

if "drawn_layers" not in st.session_state:
    st.session_state["drawn_layers"] = {}
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [-15.793889, -47.882778]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

LAYER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

def next_color():
    idx = len(st.session_state["drawn_layers"]) % len(LAYER_COLORS)
    return LAYER_COLORS[idx]

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

def find_sectors(query_text: str):

    _, by_code, by_subdist = load_geojson()
    q = normalize(query_text)

    code_match = re.search(r"\b(\d{15})\b", query_text)
    if code_match:
        code = code_match.group(1)
        if code in by_code:
            return code, [by_code[code]]

    best_key   = None
    best_score = 0
    for key in by_subdist:
        if q in key or key in q:
            score = len(key)
            if score > best_score:
                best_score = score
                best_key   = key
    if best_key:
        return best_key.title(), by_subdist[best_key]

    return None, []


def parse_map_command(user_text: str):

    subdists = get_subdist_list()
    subdist_list_str = ", ".join(subdists)

    system_prompt = f"""Você é um agente de controle de mapa para uma aplicação de saúde pública do Distrito Federal.
Sua única tarefa é detectar se o usuário quer desenhar ou remover setores censitários no mapa.

Subdistritos disponíveis (RA): {subdist_list_str}

Responda SOMENTE com JSON puro, sem markdown, no formato:
{{"action": "draw"|"remove"|"clear"|"none", "target": "<nome do subdistrito, código de setor ou 'all'>"}}

Regras:
- "draw"   → usuário quer ver/marcar/destacar/mostrar/desenhar um setor ou RA no mapa
- "remove" → usuário quer remover/apagar/limpar um setor ou RA específico
- "clear"  → usuário quer limpar TODOS os setores do mapa
- "none"   → não é um comando de mapa (é uma pergunta de saúde ou outro assunto)
- Para "target", use o nome mais próximo da lista acima, ou o código do setor, ou "all" para clear.
- Normalize variações ortográficas e abreviações para o nome correto da lista.
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    try:
        return json.loads(raw)
    except Exception:
        return {"action": "none", "target": ""}


def execute_map_command(parsed: dict) -> str:

    action = parsed.get("action", "none")
    target = parsed.get("target", "")

    if action == "clear":
        st.session_state["drawn_layers"] = {}
        return "🗺️ Todos os setores foram removidos do mapa."

    if action == "remove":
        key_norm = normalize(target)
        removed  = []
        for k in list(st.session_state["drawn_layers"].keys()):
            if normalize(k) == key_norm or key_norm in normalize(k):
                del st.session_state["drawn_layers"][k]
                removed.append(k)
        if removed:
            return f"🗑️ Camada(s) removida(s): {', '.join(removed)}."
        return f"⚠️ Nenhuma camada chamada **{target}** encontrada no mapa."

    if action == "draw":
        label, features = find_sectors(target)
        if not features:
            return (
                f"⚠️ Não encontrei setores para **{target}**. "
                f"Tente o nome de uma RA (ex: Ceilândia, Taguatinga) ou o código do setor (15 dígitos)."
            )

        color = next_color()
        st.session_state["drawn_layers"][label] = {
            "features": features,
            "color":    color,
        }

        lats, lons = [], []
        for feat in features:
            coords = feat["geometry"]["coordinates"]
            for ring in coords:
                for lon, lat in ring:
                    lats.append(lat)
                    lons.append(lon)
        if lats:
            st.session_state["map_center"] = [
                sum(lats) / len(lats),
                sum(lons) / len(lons),
            ]

        n = len(features)
        return (
            f"✅ **{label}** desenhado no mapa com {n} setor(es) censitário(s)."
        )

    return None

def answer_health_question(pergunta: str) -> str:
    documentos = retriever.invoke(pergunta)
    MAX_CHARS  = 6000
    contexto   = ""
    for doc in documentos:
        if len(contexto) + len(doc.page_content) > MAX_CHARS:
            break
        contexto += doc.page_content + "\n\n---\n\n"

    prompt = f"""Você é um assistente especializado em saúde pública brasileira.
Responda exclusivamente com base nas informações fornecidas no contexto.
Se a resposta não estiver claramente presente, diga que não encontrou evidência suficiente.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",  "content": "Você responde de forma técnica, objetiva e em português."},
            {"role": "user",    "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content

col_map, col_chat = st.columns([1, 1])

with col_map:
    st.subheader("Mapa do DF – Setores Censitários")

    layers = st.session_state["drawn_layers"]
    if layers:
        legend_html = (
            "<div style='font-size:0.82rem;margin-bottom:6px'>"
            "<b>Camadas ativas:</b> "
            + " &nbsp;".join(
                f"<span style='color:{v['color']};font-weight:600'>■ {k} "
                f"({len(v['features'])} setores)</span>"
                for k, v in layers.items()
            )
            + "</div>"
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        if st.button("🗑️ Limpar todas as camadas"):
            st.session_state["drawn_layers"] = {}
            st.rerun()

    center = st.session_state["map_center"]
    zoom   = 11 if not layers else 12

    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    for label, layer_data in layers.items():
        color    = layer_data["color"]
        features = layer_data["features"]

        geojson_payload = {
            "type":     "FeatureCollection",
            "features": features,
        }

        def make_style(c):
            def style_fn(_):
                return {
                    "fillColor":   c,
                    "color":       c,
                    "weight":      1.2,
                    "fillOpacity": 0.35,
                }
            return style_fn

        def make_tooltip(lbl):
            return folium.GeoJsonTooltip(
                fields=["CD_SETOR", "NM_SUBDIST"],
                aliases=["Setor:", "RA:"],
                localize=True,
            )

        folium.GeoJson(
            geojson_payload,
            name=label,
            style_function=make_style(color),
            tooltip=make_tooltip(label),
        ).add_to(m)


    folium.Marker(
        [-15.793889, -47.882778],
        popup="Fiocruz Brasília",
        tooltip="Fiocruz Brasília",
        icon=folium.Icon(color="red", icon="plus-sign"),
    ).add_to(m)

    if len(layers) > 1:
        folium.LayerControl().add_to(m)

    st_folium(m, width=None, height=480, returned_objects=[])


with col_chat:
    st.subheader("Assistente em Saúde Pública + Agente de Mapa")

    with st.expander("💡 Exemplos de comandos"):
        st.markdown(
            """
**Mapa – desenhar:**
- _Desenha Ceilândia no mapa_
- _Mostra os setores de Taguatinga_
- _Marca o setor 530010805060005_

**Mapa – remover:**
- _Remove Ceilândia do mapa_
- _Apaga Taguatinga_
- _Limpa tudo do mapa_

**Saúde pública (RAG):**
- _Quais são as principais doenças notificadas no DF?_
- _Como está a cobertura vacinal?_
"""
        )

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            role_label = "🧑 Você" if msg["role"] == "user" else "🤖 Agente"
            st.markdown(f"**{role_label}:** {msg['content']}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Digite uma pergunta ou comando de mapa:",
            placeholder="Ex: Desenha Samambaia | Quais doenças são mais comuns no DF?",
        )
        submitted = st.form_submit_button("Enviar")

    if submitted and user_input.strip():
        user_msg = user_input.strip()
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_msg}
        )

        with st.spinner("Processando..."):

            parsed   = parse_map_command(user_msg)
            response = execute_map_command(parsed)

            if response is None:
                response = answer_health_question(user_msg)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

    if st.session_state["chat_history"]:
        if st.button("Limpar conversa"):
            st.session_state["chat_history"] = []
            st.rerun()


st.markdown("---")
st.markdown(
    "© 2026 – **Saruê** – Fiocruz Brasília  \n"
    "Grupo de Inteligência Computacional na Saúde (GICS)"
)
