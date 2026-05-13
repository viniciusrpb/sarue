import streamlit as st
import pandas as pd
import os
import json
import re
import unicodedata
import requests
from groq import Groq
import folium
from streamlit_folium import st_folium
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="Opossum ::: Fiocruz Brasília", layout="wide")
st.title("Opossum ::: Fiocruz Brasília")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

UBS_CATEGORIES = {"ubs", "ups", "unidade basica", "unidade básica"}

HEADERS_OSM = {"User-Agent": "Sarue-Fiocruz/2.0 (fiocruz.br)"}

RISCO_COLORS = {"Muito Alto": "#d73027", "Alto": "#fc8d59"}

QUEIMADA_COLORS = {
    3: "#fee08b", 6: "#fdae61", 7: "#f46d43",
    8: "#d73027", 9: "#a50026", 10: "#7f0000",
}

RA_BBOX = {
    "asa norte":          "-15.775,-47.920,-15.710,-47.870",
    "asa sul":            "-15.840,-47.930,-15.775,-47.870",
    "taguatinga":         "-15.870,-48.080,-15.790,-48.020",
    "ceilandia":          "-15.850,-48.130,-15.760,-48.060",
    "samambaia":          "-15.900,-48.120,-15.840,-48.050",
    "gama":               "-16.060,-48.090,-15.990,-47.990",
    "sobradinho":         "-15.660,-47.870,-15.590,-47.790",
    "planaltina":         "-15.650,-47.680,-15.560,-47.590",
    "brazlandia":         "-15.730,-48.230,-15.650,-48.140",
    "paranoa":            "-15.740,-47.780,-15.660,-47.720",
    "nucleo bandeirante": "-15.880,-47.990,-15.840,-47.940",
    "guara":              "-15.840,-47.990,-15.790,-47.940",
    "cruzeiro":           "-15.800,-47.970,-15.760,-47.930",
    "lago norte":         "-15.730,-47.880,-15.680,-47.820",
    "lago sul":           "-15.870,-47.870,-15.820,-47.810",
    "aguas claras":       "-15.880,-48.040,-15.840,-47.990",
    "riacho fundo":       "-15.910,-48.040,-15.870,-47.990",
    "vicente pires":      "-15.840,-48.070,-15.790,-48.020",
    "itapoa":             "-15.720,-47.780,-15.670,-47.740",
    "estrutural":         "-15.790,-48.030,-15.760,-47.990",
}

OSM_CATEGORIES = {
    "hospital":       ("amenity", "hospital",        "🏥", "red"),
    "ubs":            ("amenity", "clinic",           "🏥", "blue"),
    "ups":            ("amenity", "clinic",           "🏥", "blue"),
    "clinica":        ("amenity", "clinic",           "🏥", "blue"),
    "clínica":        ("amenity", "clinic",           "🏥", "blue"),
    "clinic":         ("amenity", "clinic",           "🏥", "blue"),
    "farmacia":       ("amenity", "pharmacy",         "💊", "green"),
    "farmácia":       ("amenity", "pharmacy",         "💊", "green"),
    "pharmacy":       ("amenity", "pharmacy",         "💊", "green"),
    "medico":         ("amenity", "doctors",          "👨‍⚕️", "cadetblue"),
    "médico":         ("amenity", "doctors",          "👨‍⚕️", "cadetblue"),
    "doctor":         ("amenity", "doctors",          "👨‍⚕️", "cadetblue"),
    "dentista":       ("amenity", "dentist",          "🦷", "purple"),
    "dentist":        ("amenity", "dentist",          "🦷", "purple"),
    "social":         ("amenity", "social_facility",  "🤝", "orange"),
    "cras":           ("amenity", "social_facility",  "🤝", "orange"),
    "creas":          ("amenity", "social_facility",  "🤝", "orange"),
    "social facility":("amenity", "social_facility",  "🤝", "orange"),
    "escola":         ("amenity", "school",           "🏫", "darkblue"),
    "school":         ("amenity", "school",           "🏫", "darkblue"),
    "creche":         ("amenity", "kindergarten",     "👶", "pink"),
    "kindergarten":   ("amenity", "kindergarten",     "👶", "pink"),
    "saude":          ("healthcare", "*",             "➕", "red"),
    "saúde":          ("healthcare", "*",             "➕", "red"),
    "health":         ("healthcare", "*",             "➕", "red"),
    "healthcare":     ("healthcare", "*",             "➕", "red"),
    "academia":       ("leisure", "fitness_centre",   "🏋️", "darkgreen"),
    "gym":            ("leisure", "fitness_centre",   "🏋️", "darkgreen"),
    "fitness":        ("leisure", "fitness_centre",   "🏋️", "darkgreen"),
}

DF_BBOX = "-16.05,-48.28,-15.48,-47.30"

POLY_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

def _norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

@st.cache_data(show_spinner=False)
def load_ubs_df():
    path = os.path.join(os.path.dirname(__file__), "database", "ubs_df.csv")
    df = pd.read_csv(path, dtype=str)
    for col in ["latitude", "longitude"]:
        df[col] = df[col].str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df["nome_norm"]   = df["nome"].apply(_norm)
    df["bairro_norm"] = df["bairro"].apply(_norm)
    return df


@st.cache_data(show_spinner=False)
def load_geojson():
    path = os.path.join(os.path.dirname(__file__), "database", "setoresDF.json")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    by_code, by_subdist = {}, {}
    for feat in gj["features"]:
        props = feat["properties"]
        code  = props.get("CD_SETOR", "")
        sd    = _norm(props.get("NM_SUBDIST") or "")
        by_code[code] = feat
        by_subdist.setdefault(sd, []).append(feat)
    return gj, by_code, by_subdist


@st.cache_data(show_spinner=False)
def get_subdist_list():
    _, _, by_subdist = load_geojson()
    return sorted(by_subdist.keys())


@st.cache_data(show_spinner=False)
def load_documents():
    base_dir  = os.path.join(os.path.dirname(__file__), "database", "news")
    documents = []
    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(base_dir, filename)
        try:
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            continue
        for item in items:
            titulo     = str(item.get("title",        "")).strip()
            conteudo   = str(item.get("content",      "")).strip()
            url        = str(item.get("url",          "")).strip()
            localidades = str(item.get("localidades", "")).strip()
            tipos      = str(item.get("tipos_unidade","")).strip()
            text = f"{titulo}\n\n{conteudo}".strip()
            if len(text) < 50:
                continue
            documents.append(Document(
                page_content=text,
                metadata={"source": filename, "url": url,
                          "localidades": localidades, "tipos": tipos},
            ))
    return documents


@st.cache_data(show_spinner=False)
def load_dengue_data():
    path = os.path.join(os.path.dirname(__file__), "database",
                        "dados_dengue-16042026-ano_2026.csv")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_risco_geologico():
    path = os.path.join(os.path.dirname(__file__), "database", "risco_geologico.geojson")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_queimadas():
    path = os.path.join(os.path.dirname(__file__), "database", "area_queimada_2025.geojson")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def aggregate_dengue_by_sector(df):
    return df.groupby("cd_setor")["casos"].sum().reset_index()


@st.cache_data(show_spinner=False)
def attach_dengue_to_geojson():
    gj, by_code, _ = load_geojson()
    df  = load_dengue_data()
    agg = aggregate_dengue_by_sector(df)
    dengue_map = dict(zip(agg["cd_setor"], agg["casos"]))
    for feat in gj["features"]:
        code = feat["properties"].get("CD_SETOR")
        feat["properties"]["dengue_casos"] = int(dengue_map.get(code, 0))
    return gj


@st.cache_data(show_spinner=False)
def compute_dengue_breaks():
    df     = load_dengue_data()
    agg    = aggregate_dengue_by_sector(df)
    valores = agg["casos"][agg["casos"] > 0]
    if valores.empty:
        return [1, 5, 10, 25, 50, 100]
    breaks = [
        int(valores.quantile(0.50)),
        int(valores.quantile(0.65)),
        int(valores.quantile(0.75)),
        int(valores.quantile(0.85)),
        int(valores.quantile(0.92)),
        int(valores.quantile(0.97)),
    ]
    seen = []
    for b in breaks:
        if not seen or b > seen[-1]:
            seen.append(b)
    return seen


def get_dengue_color(value):
    if value == 0:
        return "#FFEDA0"
    breaks = compute_dengue_breaks()
    colors = ["#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
    for i, b in enumerate(breaks):
        if value <= b:
            return colors[min(i, len(colors) - 1)]
    return colors[-1]

@st.cache_data(show_spinner=False, ttl=300)
@st.cache_data(show_spinner=False, ttl=300)
def extract_entities(text):
    resp = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract named entities from the user query. "
                    "Reply ONLY with pure JSON, no markdown, no thinking, no explanation.\n"
                    "Format: {\"locations\": [], \"organizations\": [], \"events\": []}"
                ),
            },
            {"role": "user", "content": f"/no_think {text}"},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    try:
        entities = json.loads(raw)
        return {
            "locations":     [_norm(e) for e in entities.get("locations",     [])],
            "organizations": [_norm(e) for e in entities.get("organizations", [])],
            "events":        [_norm(e) for e in entities.get("events",        [])],
        }
    except Exception:
        return {"locations": [], "organizations": [], "events": []}


@st.cache_resource(show_spinner=False)
def setup_retriever():
    docs = load_documents()
    enriched = []
    for doc in docs:
        locs  = doc.metadata.get("localidades", "").replace(";", " ")
        tipos = doc.metadata.get("tipos", "").replace(";", " ")
        prefix = f"[local: {locs}] [tipo: {tipos}]\n" if (locs or tipos) else ""
        enriched.append(Document(
            page_content=prefix + doc.page_content,
            metadata=doc.metadata,
        ))
    splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separator="\n\n")
    chunks   = splitter.split_documents(enriched)
    embeddings = HuggingFaceEmbeddings(
        model_name="alfaneo/bertimbau-base-portuguese-sts",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    dense_db       = FAISS.from_documents(chunks, embedding=embeddings)
    dense_retriever = dense_db.as_retriever(search_kwargs={"k": 12})
    corpus = [c.page_content.lower().split() for c in chunks]
    bm25   = BM25Okapi(corpus)
    return dense_retriever, bm25, chunks


def _rerank(dense_docs, bm25_docs, entities, top_k=6):
    all_entities = (
        entities["locations"] + entities["organizations"] + entities["events"]
    )
    scored = {}
    for rank, doc in enumerate(dense_docs):
        key = id(doc)
        scored[key] = {"doc": doc, "score": len(dense_docs) - rank}
    for rank, doc in enumerate(bm25_docs):
        key = id(doc)
        if key in scored:
            scored[key]["score"] += (len(bm25_docs) - rank) * 0.5
        else:
            scored[key] = {"doc": doc, "score": (len(bm25_docs) - rank) * 0.5}
    for item in scored.values():
        doc  = item["doc"]
        text = (
            _norm(doc.metadata.get("localidades", "")) + " " +
            _norm(doc.metadata.get("tipos",        "")) + " " +
            _norm(doc.page_content[:300])
        )
        bonus = sum(2.0 for e in all_entities if e and e in text)
        item["score"] += bonus
    ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked[:top_k]]


def answer_health_question(pergunta, lang="en"):
    dense_retriever, bm25, chunks = setup_retriever()
    entities   = extract_entities(pergunta)
    dense_docs = dense_retriever.invoke(pergunta)

    query_terms = pergunta.lower().split()
    for e in entities["locations"] + entities["organizations"] + entities["events"]:
        query_terms += e.split()
    bm25_scores  = bm25.get_scores(query_terms)
    top_bm25_idx = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:12]
    bm25_docs  = [chunks[i] for i in top_bm25_idx]
    final_docs = _rerank(dense_docs, bm25_docs, entities, top_k=6)

    MAX_CHARS = 6000
    contexto  = ""
    localidades_encontradas = set()
    for doc in final_docs:
        if len(contexto) + len(doc.page_content) > MAX_CHARS:
            break
        contexto += doc.page_content + "\n\n---\n\n"
        for loc in doc.metadata.get("localidades", "").split(";"):
            loc = loc.strip()
            if loc:
                localidades_encontradas.add(loc)

    lang_instruction = (
        "Answer in English. Translate relevant Portuguese content as needed."
        if lang == "en"
        else "Responda em português."
    )
    entity_hint = ""
    if any(entities.values()):
        flat = ", ".join(
            entities["locations"] + entities["organizations"] + entities["events"]
        )
        entity_hint = f"Key entities detected in the query: {flat}\n"

    prompt = f"""You are an assistant specialised in Brazilian public health (Distrito Federal).
Answer based on the news articles in the context below.
{lang_instruction}
{entity_hint}
Rules:
- Answer directly — never say "according to the context".
- Cite the source as "Secretaria de Saúde do Distrito Federal (SES-DF)".
- Include dates, times, and locations when present in the context.
- If truly nothing is found, say so briefly.

Context:
{contexto}

Question:
{pergunta}

Answer:"""

    resp = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": "You are a concise, factual public-health assistant. Do not think out loud."},
            {"role": "user",   "content": f"/no_think {prompt}"},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw, localidades_encontradas

@st.cache_data(show_spinner=False, ttl=86400)
def get_area_bbox(area_name):
    params = {
        "q": f"{area_name}, Distrito Federal, Brasil",
        "format": "json", "limit": 1, "addressdetails": 0,
        "countrycodes": "br",
        "viewbox": "-48.28,-16.05,-47.30,-15.48", "bounded": 1,
    }
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params=params, headers=HEADERS_OSM, timeout=10)
        r.raise_for_status()
        results = r.json()
        if results and "boundingbox" in results[0]:
            bb = results[0]["boundingbox"]
            return f"{bb[0]},{bb[2]},{bb[1]},{bb[3]}"
    except Exception:
        pass
    return None


def geocode(address):
    params = {
        "q": f"{address}, Distrito Federal, Brasil",
        "format": "json", "limit": 5, "addressdetails": 1,
        "countrycodes": "br",
        "viewbox": "-48.28,-16.05,-47.30,-15.48", "bounded": 1,
    }
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params=params, headers=HEADERS_OSM, timeout=10)
        r.raise_for_status()
        return [
            {"display_name": d["display_name"], "lat": float(d["lat"]),
             "lon": float(d["lon"]), "osm_type": d.get("type", "")}
            for d in r.json()
        ], None
    except requests.exceptions.Timeout:
        return [], "Connection timeout. Please try again."
    except Exception as e:
        return [], str(e)


def _overpass_query(key, value, area_name, name_filter=None):
    name_clause = f'["name"~"{name_filter}",i]' if name_filter else ""
    bbox = DF_BBOX
    if area_name:
        area_bbox = get_area_bbox(area_name)
        if area_bbox:
            bbox = area_bbox
    if value == "*":
        selectors = f'node["{key}"]{name_clause}({bbox}); way["{key}"]{name_clause}({bbox});'
    else:
        selectors = f'node["{key}"="{value}"]{name_clause}({bbox}); way["{key}"="{value}"]{name_clause}({bbox});'
    return f"[out:json][timeout:25];\n({selectors});\nout center 100;"


import time

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

@st.cache_data(show_spinner=False, ttl=3600)
def _overpass_request(query):
    last_err = None
    for i, url in enumerate(OVERPASS_SERVERS):
        try:
            if i > 0:
                time.sleep(3)
            r = requests.post(url, data={"data": query},
                              headers=HEADERS_OSM, timeout=60)
            r.raise_for_status()
            return r.json().get("elements", []), None
        except requests.exceptions.Timeout:
            last_err = f"Timeout at {url}"
        except Exception as e:
            last_err = str(e)
    return [], last_err


def search_ubs_local(area_name=None, name_filter=None):
    df = load_ubs_df()
    if area_name:
        area_n = _norm(area_name)
        df = df[
            df["bairro_norm"].str.contains(area_n, na=False) |
            df["nome_norm"].str.contains(area_n, na=False)
        ]
    if name_filter:
        df = df[df["nome_norm"].str.contains(_norm(name_filter), na=False)]
    return [
        {"name": row["nome"].title(), "lat": row["latitude"],
         "lon": row["longitude"], "address": row.get("logradouro", ""), "phone": ""}
        for _, row in df.iterrows()
    ]


def search_poi(category, area_name, name_filter=None):
    cat_norm = _norm(category)
    if any(cat_norm == _norm(k) or _norm(k) in cat_norm for k in UBS_CATEGORIES):
        pois = search_ubs_local(area_name=area_name, name_filter=name_filter)
        if pois:
            return pois, "🏥", "blue", None

    matched = None
    for k, v in OSM_CATEGORIES.items():
        if cat_norm == _norm(k) or _norm(k) in cat_norm or cat_norm in _norm(k):
            matched = v
            break
    if matched:
        osm_key, osm_val, icon, color = matched
    else:
        osm_key, osm_val, icon, color = "amenity", cat_norm, "📍", "gray"

    query    = _overpass_query(osm_key, osm_val, area_name, name_filter=name_filter)
    elements, err = _overpass_request(query)
    if err:
        return [], icon, color, err

    pois = []
    for el in elements:
        tags = el.get("tags", {})
        name = (tags.get("name") or tags.get("name:pt")
                or tags.get("operator") or osm_val.title())
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")
        if lat and lon:
            pois.append({"name": name, "lat": lat, "lon": lon,
                         "address": tags.get("addr:street", ""),
                         "phone": tags.get("phone", tags.get("contact:phone", ""))})
    return pois, icon, color, None


# ── Mapa ──────────────────────────────────────────────────────────────────────

def next_poly_color():
    return POLY_COLORS[len(st.session_state["drawn_layers"]) % len(POLY_COLORS)]


def find_sectors(query_text):
    _, by_code, by_subdist = load_geojson()
    q = _norm(query_text)
    code_match = re.search(r"\b(\d{15})\b", query_text)
    if code_match:
        code = code_match.group(1)
        if code in by_code:
            return code, [by_code[code]]
    best_key, best_score = None, 0
    for key in by_subdist:
        if q in key or key in q:
            score = len(key)
            if score > best_score:
                best_score, best_key = score, key
    if best_key:
        return best_key.title(), by_subdist[best_key]
    return None, []


def detect_language(text):
    resp = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Detect the language of the user message. "
                    "Reply with only 'pt' if it is Portuguese, or 'en' if it is English or any other language. "
                    "Do not explain or think out loud."
                ),
            },
            {"role": "user", "content": f"/no_think {text}"},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    lang = resp.choices[0].message.content.strip().lower()
    # remove qualquer <think>...</think> que vaze
    lang = re.sub(r"<think>.*?</think>", "", lang, flags=re.DOTALL).strip()
    return "pt" if "pt" in lang else "en"


def parse_command(user_text):
    subdist_str  = ", ".join(get_subdist_list())
    poi_cats_str = ", ".join(sorted({_norm(k) for k in OSM_CATEGORIES}))

    system = f"""You are the map control agent for the Opossum app (Fiocruz Brasília, DF, Brazil).
        Classify the user message and respond ONLY with pure JSON (no markdown, no explanation).

        Actions:
        - "draw":     draw census sector polygons for an Administrative Region (RA) on the map
        - "remove":   remove a specific layer from the map
        - "clear":    remove ALL layers from the map
        - "poi":      search for points of interest on OpenStreetMap (hospitals, pharmacies, clinics, etc.)
        - "geocode":  locate and pin a specific address or place on the map
        - "dengue":   display a dengue case choropleth map by census sector
        - "risco":    display geological risk areas on the map (CPRM data)
        - "queimada": display burned areas on the map (2025 data)
        - "none":     public health question or topic unrelated to map control

        Available RAs (for draw/remove, Portuguese names): {subdist_str}
        Available POI categories: {poi_cats_str}

        Response format:
        {{
        "action":      "draw"|"remove"|"clear"|"poi"|"geocode"|"dengue"|"risco"|"queimada"|"none",
        "target":      "<RA name, sector code, full address or category>",
        "area":        "<RA/neighbourhood name for POI search, or null>",
        "category":    "<normalised POI category without accents, or null>",
        "name_filter": "<specific name/number to filter, e.g. '01', 'Asa Norte', or null>"
        }}

        Rules:
        - For "poi": if the user mentions a specific unit name or number (e.g. "UBS 01"), set "name_filter".
        - For "geocode": "target" = full address or place name.
        - For "draw"/"remove": "target" = RA name (normalised to the list above).
        - If ambiguous between "poi" and "draw", prefer "poi".
        - The user may write in Portuguese or English; handle both.
        """
    resp = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"/no_think {user_text}"},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    raw = resp.choices[0].message.content.strip()  # <- remova o #
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    try:
        return json.loads(raw)
    except Exception:
        return {"action": "none", "target": "", "area": None, "category": None}


def execute_command(parsed, lang="en"):
    action   = parsed.get("action",   "none")
    target   = parsed.get("target",   "") or ""
    area     = parsed.get("area",     None)
    category = parsed.get("category", None) or target

    def msg(en, pt):
        return pt if lang == "pt" else en

    if action == "clear":
        for store in ["drawn_layers", "poi_layers", "pin_layers"]:
            st.session_state[store] = {}
        st.session_state.pop("dengue_layer",  None)
        st.session_state.pop("risco_layer",   None)
        st.session_state.pop("queimada_layer", None)
        return msg("🗺️ All layers have been removed from the map.",
                   "🗺️ Todas as camadas foram removidas do mapa.")

    if action == "remove":
        key_norm = _norm(target)
        removed  = []
        for store in ["drawn_layers", "poi_layers", "pin_layers"]:
            for k in list(st.session_state[store].keys()):
                if _norm(k) == key_norm or key_norm in _norm(k):
                    del st.session_state[store][k]
                    removed.append(k)
        for special in ["dengue_layer", "risco_layer", "queimada_layer"]:
            label_map = {"dengue_layer": "dengue", "risco_layer": "risco",
                         "queimada_layer": "queimada"}
            if key_norm in label_map.get(special, "") and special in st.session_state:
                del st.session_state[special]
                removed.append(special)
        if removed:
            return msg(f"🗑️ Layer(s) removed: {', '.join(removed)}.",
                       f"🗑️ Camada(s) removida(s): {', '.join(removed)}.")
        return msg(f"⚠️ No layer named **{target}** found on the map.",
                   f"⚠️ Nenhuma camada chamada **{target}** encontrada no mapa.")

    if action == "draw":
        label, features = find_sectors(target)
        if not features:
            return msg(
                f"⚠️ No sectors found for **{target}**.\n"
                "Try the name of an Administrative Region (e.g. Ceilândia, Taguatinga) "
                "or a 15-digit sector code.",
                f"⚠️ Não encontrei setores para **{target}**.\n"
                "Tente o nome de uma RA (ex: Ceilândia, Taguatinga) "
                "ou um código de setor com 15 dígitos.",
            )
        color = next_poly_color()
        st.session_state["drawn_layers"][label] = {"features": features, "color": color}
        lats, lons = [], []
        for feat in features:
            for ring in feat["geometry"]["coordinates"]:
                for lon, lat in ring:
                    lats.append(lat); lons.append(lon)
        if lats:
            st.session_state["map_center"] = [
                sum(lats) / len(lats), sum(lons) / len(lons)]
        return msg(f"✅ **{label}** drawn on the map with {len(features)} census sector(s).",
                   f"✅ **{label}** desenhado no mapa com {len(features)} setor(es) censitário(s).")

    if action == "poi":
        name_filter = parsed.get("name_filter", None)
        with st.spinner(f"Querying OpenStreetMap: **{category}**..."):
            pois, icon, color, err = search_poi(category, area, name_filter=name_filter)
        if err:
            return msg(f"⚠️ Error querying the Overpass API: `{err}`",
                       f"⚠️ Erro ao consultar a Overpass API: `{err}`")
        if not pois:
            area_msg_en = f" in **{area}**" if area else " in the Federal District"
            area_msg_pt = f" em **{area}**" if area else " no DF"
            return msg(
                f"🔍 No results found for **{category}**{area_msg_en}.\n"
                "Try another category (e.g. hospital, pharmacy, clinic, ubs).",
                f"🔍 Nenhum resultado encontrado para **{category}**{area_msg_pt}.\n"
                "Tente outra categoria (ex: hospital, farmacia, clinica, ubs).",
            )
        layer_label = f"{icon} {category.title()}"
        if area:
            layer_label += f" – {area.title()}"
        st.session_state["poi_layers"][layer_label] = {"pois": pois, "icon": icon, "color": color}
        lats = [p["lat"] for p in pois]
        lons = [p["lon"] for p in pois]
        st.session_state["map_center"] = [sum(lats) / len(lats), sum(lons) / len(lons)]
        area_msg_en = f" in **{area.title()}**" if area else " in the Federal District"
        area_msg_pt = f" em **{area.title()}**" if area else " no DF"
        lista_txt = "\n".join(
            f"- **{p['name']}**"
            + (f" — {p['address']}" if p["address"] else "")
            + (f" ☎ {p['phone']}"   if p["phone"]   else "")
            for p in pois[:10]
        )
        suffix_en = "\n\n*(showing first 10 results)*" if len(pois) > 10 else ""
        suffix_pt = "\n\n*(exibindo os primeiros 10)*"  if len(pois) > 10 else ""
        return msg(
            f"📍 **{len(pois)} result(s)** for **{category.title()}**{area_msg_en}:\n\n"
            + lista_txt + suffix_en,
            f"📍 **{len(pois)} resultado(s)** para **{category.title()}**{area_msg_pt}:\n\n"
            + lista_txt + suffix_pt,
        )

    if action == "dengue":
        gj = attach_dengue_to_geojson()
        st.session_state["dengue_layer"] = gj
        st.session_state["map_center"]   = [-15.793889, -47.882778]
        return msg("🦟 Dengue case map loaded by census sector.",
                   "🦟 Mapa de casos de dengue carregado por setor censitário.")

    if action == "risco":
        gj = load_risco_geologico()
        st.session_state["risco_layer"] = gj
        st.session_state["map_center"]  = [-15.793889, -47.882778]
        n = len(gj["features"])
        return msg(f"⚠️ {n} geological risk sectors loaded (Alto and Muito Alto).",
                   f"⚠️ {n} setores de risco geológico carregados (Alto e Muito Alto).")

    if action == "queimada":
        gj = load_queimadas()
        st.session_state["queimada_layer"] = gj
        st.session_state["map_center"]     = [-15.793889, -47.882778]
        n = len(gj["features"])
        return msg(f"🔥 {n} burned area polygons loaded (2025 data).",
                   f"🔥 {n} polígonos de áreas queimadas carregados (dados 2025).")

    if action == "geocode":
        with st.spinner(f"Locating **{target}**..."):
            results, err = geocode(target)
        if err:
            return msg(f"⚠️ Error querying Nominatim: `{err}`",
                       f"⚠️ Erro ao consultar o Nominatim: `{err}`")
        if not results:
            return msg(
                f"⚠️ Could not locate **{target}** in the Federal District.\n"
                "Try including the Administrative Region name or a more complete address.",
                f"⚠️ Não foi possível localizar **{target}** no DF.\n"
                "Tente incluir o nome da RA ou um endereço mais completo.",
            )
        best  = results[0]
        label = target.title()
        st.session_state["pin_layers"][label] = {
            "lat": best["lat"], "lon": best["lon"],
            "display_name": best["display_name"],
        }
        st.session_state["map_center"] = [best["lat"], best["lon"]]
        short_name = best["display_name"].split(",")[0]
        return msg(
            f"📌 **{short_name}** located and pinned on the map.\n"
            f"Coordinates: `{best['lat']:.5f}, {best['lon']:.5f}`",
            f"📌 **{short_name}** localizado e marcado no mapa.\n"
            f"Coordenadas: `{best['lat']:.5f}, {best['lon']:.5f}`",
        )

    return None


# ── Session state ─────────────────────────────────────────────────────────────

if "drawn_layers"  not in st.session_state: st.session_state["drawn_layers"]  = {}
if "poi_layers"    not in st.session_state: st.session_state["poi_layers"]    = {}
if "pin_layers"    not in st.session_state: st.session_state["pin_layers"]    = {}
if "map_center"    not in st.session_state: st.session_state["map_center"]    = [-15.793889, -47.882778]
if "chat_history"  not in st.session_state: st.session_state["chat_history"]  = []


# ── Layout ────────────────────────────────────────────────────────────────────

col_map, col_chat = st.columns([1, 1])

with col_map:
    st.subheader("Federal District (DF)")

    all_labels = (
        list(st.session_state["drawn_layers"].keys())
        + list(st.session_state["poi_layers"].keys())
        + list(st.session_state["pin_layers"].keys())
    )
    special_labels = [
        k for k in ("dengue_layer", "risco_layer", "queimada_layer")
        if k in st.session_state
    ]
    display_labels = all_labels + [s.replace("_layer", "") for s in special_labels]

    if display_labels:
        st.caption("**Active layers:** " + " · ".join(display_labels))
        if st.button("🗑️ Clear all layers"):
            for store in ["drawn_layers", "poi_layers", "pin_layers"]:
                st.session_state[store] = {}
            for k in ["dengue_layer", "risco_layer", "queimada_layer"]:
                st.session_state.pop(k, None)
            st.rerun()

    center = st.session_state["map_center"]
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    folium.TileLayer(tiles="CartoDB positron",  name="Street map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Relief", control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Satellite", control=True,
    ).add_to(m)

    # Dengue
    if "dengue_layer" in st.session_state:
        folium.GeoJson(
            st.session_state["dengue_layer"],
            name="Dengue 2026",
            style_function=lambda feature: {
                "fillColor": get_dengue_color(feature["properties"]["dengue_casos"]),
                "color": "black", "weight": 0.3, "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NM_SUBDIST", "dengue_casos"],
                aliases=["Região:", "Casos:"],
            ),
        ).add_to(m)

    # Risco geológico
    if "risco_layer" in st.session_state:
        folium.GeoJson(
            st.session_state["risco_layer"],
            name="Risco Geológico (CPRM)",
            style_function=lambda feat: {
                "fillColor":   RISCO_COLORS.get(feat["properties"]["grau_risco"], "#fc8d59"),
                "color": "#333", "weight": 1.5, "fillOpacity": 0.65,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["local", "tipologia", "grau_risco", "situacao", "num_edif", "num_pess"],
                aliases=["Local:", "Tipologia:", "Grau de risco:", "Situação:", "Edificações:", "Pessoas:"],
                localize=True,
            ),
            popup=folium.GeoJsonPopup(
                fields=["local", "tipologia", "grau_risco", "situacao",
                        "num_edif", "num_pess", "descricao", "sug_interv"],
                aliases=["Local", "Tipologia", "Grau de risco", "Situação",
                         "Edificações", "Pessoas", "Descrição", "Intervenção sugerida"],
                max_width=400,
            ),
        ).add_to(m)

    # Áreas queimadas
    if "queimada_layer" in st.session_state:
        folium.GeoJson(
            st.session_state["queimada_layer"],
            name="Áreas Queimadas 2025",
            style_function=lambda feat: {
                "fillColor":   QUEIMADA_COLORS.get(feat["properties"]["mes"], "#fdae61"),
                "color": "#333", "weight": 0.5, "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["mes_nome", "area_ha", "data"],
                aliases=["Mês:", "Área (ha):", "Data:"],
            ),
        ).add_to(m)

    # Setores censitários
    for label, layer in st.session_state["drawn_layers"].items():
        color   = layer["color"]
        payload = {"type": "FeatureCollection", "features": layer["features"]}
        folium.GeoJson(
            payload, name=label,
            style_function=lambda _, c=color: {
                "fillColor": c, "color": c, "weight": 1.2, "fillOpacity": 0.30,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["CD_SETOR", "NM_SUBDIST"],
                aliases=["Setor:", "Região:"],
            ),
        ).add_to(m)

    # POIs
    for label, layer in st.session_state["poi_layers"].items():
        fg = folium.FeatureGroup(name=label)
        for poi in layer["pois"]:
            popup_html = (
                f"<b>{poi['name']}</b>"
                + (f"<br>{poi['address']}" if poi["address"] else "")
                + (f"<br>☎ {poi['phone']}"  if poi["phone"]   else "")
            )
            folium.Marker(
                location=[poi["lat"], poi["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=poi["name"],
                icon=folium.Icon(color=layer["color"], icon="plus-sign"),
            ).add_to(fg)
        fg.add_to(m)

    # Pins geocode
    for label, pin in st.session_state["pin_layers"].items():
        folium.Marker(
            location=[pin["lat"], pin["lon"]],
            popup=folium.Popup(pin["display_name"], max_width=300),
            tooltip=label,
            icon=folium.Icon(color="darkred", icon="map-marker"),
        ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    st_folium(m, width=None, height=500, returned_objects=[])


with col_chat:
    st.subheader("Assistant")

    for msg_item in st.session_state["chat_history"]:
        role_label = "You" if msg_item["role"] == "user" else "Opossum"
        with st.chat_message(msg_item["role"]):
            st.markdown(f"**{role_label}:** {msg_item['content']}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type a command or question:",
            placeholder="e.g. Which UBSs were contemplated with new vaccines? | Which regions are within risky regions?",
        )
        submitted = st.form_submit_button("Send ↩")

    if submitted and user_input.strip():
        user_msg = user_input.strip()
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})

        with st.spinner("Processing..."):
            lang     = detect_language(user_msg)
            parsed   = parse_command(user_msg)
            response = execute_command(parsed, lang=lang)

            if response is None:
                response, localidades = answer_health_question(user_msg, lang=lang)
                entities = extract_entities(user_msg)

                # filtra localidades: só desenha as que batem com entidades da pergunta
                entidades_loc = set(entities["locations"])
                if entidades_loc:
                    localidades = {
                        loc for loc in localidades
                        if any(e in _norm(loc) or _norm(loc) in e for e in entidades_loc)
                    }

                last_lats, last_lons = [], []
                for loc in localidades:
                    label, features = find_sectors(loc)
                    if features:
                        color = next_poly_color()
                        st.session_state["drawn_layers"][label] = {
                            "features": features,
                            "color": color,
                        }
                        for feat in features:
                            for ring in feat["geometry"]["coordinates"]:
                                for lon, lat in ring:
                                    last_lats.append(lat)
                                    last_lons.append(lon)

                if last_lats:
                    st.session_state["map_center"] = [
                        sum(last_lats) / len(last_lats),
                        sum(last_lons) / len(last_lons),
                    ]

                    st.session_state["chat_history"].append({"role": "assistant", "content": response})
                    st.rerun()

    if st.session_state["chat_history"]:
        if st.button("🧹 Clear conversation"):
            st.session_state["chat_history"] = []
            st.rerun()

st.markdown("---")
st.markdown("© 2026 · Opossum – Fiocruz Brasília · Grupo de Inteligência Computacional na Saúde (GICS)")
