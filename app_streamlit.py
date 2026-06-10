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

st.set_page_config(page_title="PICAPS/Fiocruz Brasilia :: Agent", layout="wide")
st.title("PICAPS/Fiocruz Brasilia :: Agent")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

UBS_CATEGORIES = {"ubs", "ups", "unidade basica", "unidade básica"}

HEADERS_OSM = {"User-Agent": "Sarue-Fiocruz/2.0 (fiocruz.br)"}

RISCO_COLORS = {"Muito Alto": "#d73027", "Alto": "#fc8d59"}

QUEIMADA_COLORS = {
    3: "#fee08b", 6: "#fdae61", 7: "#f46d43",
    8: "#d73027", 9: "#a50026", 10: "#7f0000",
}

RA_BBOX = {
    "arniqueira":              "-15.878,-48.033,-15.841,-47.987",
    "brazlândia":              "-15.771,-48.241,-15.502,-48.017",
    "candangolândia":          "-15.875,-47.959,-15.841,-47.932",
    "ceilândia":               "-15.933,-48.286,-15.748,-48.088",
    "cruzeiro":                "-15.806,-47.946,-15.777,-47.929",
    "fercal":                  "-15.620,-47.976,-15.502,-47.761",
    "gama":                    "-16.050,-48.279,-15.932,-48.005",
    "guará":                   "-15.864,-48.005,-15.800,-47.949",
    "itapoã":                  "-15.763,-47.792,-15.713,-47.675",
    "jardim botânico":         "-16.050,-47.908,-15.796,-47.707",
    "lago norte":              "-15.791,-47.923,-15.681,-47.784",
    "lago sul":                "-15.914,-47.956,-15.792,-47.786",
    "núcleo bandeirante":      "-15.884,-48.000,-15.848,-47.955",
    "paranoá":                 "-16.050,-47.813,-15.727,-47.308",
    "park way":                "-15.984,-48.021,-15.818,-47.887",
    "planaltina":              "-15.873,-47.781,-15.502,-47.312",
    "plano piloto":            "-15.859,-48.090,-15.578,-47.784",
    "recanto das emas":        "-15.978,-48.261,-15.882,-48.038",
    "riacho fundo":            "-15.925,-48.043,-15.874,-47.979",
    "riacho fundo ii":         "-15.966,-48.060,-15.874,-47.983",
    "scia":                    "-15.796,-48.007,-15.754,-47.966",
    "sia":                     "-15.810,-47.992,-15.740,-47.917",
    "samambaia":               "-15.937,-48.259,-15.844,-48.037",
    "santa maria":             "-16.050,-48.054,-15.969,-47.869",
    "sobradinho":              "-15.734,-47.859,-15.523,-47.673",
    "sobradinho ii":           "-15.689,-48.044,-15.502,-47.809",
    "sol nascente/pôr do sol": "-15.868,-48.191,-15.794,-48.107",
    "sudoeste/octogonal":      "-15.811,-47.949,-15.780,-47.908",
    "são sebastião":           "-16.050,-47.807,-15.875,-47.588",
    "taguatinga":              "-15.876,-48.112,-15.737,-48.026",
    "varjão":                  "-15.719,-47.892,-15.701,-47.868",
    "vicente pires":           "-15.832,-48.056,-15.752,-47.990",
    "águas claras":            "-15.851,-48.050,-15.815,-48.001",
}

def _bbox_center(bbox_str):
    """Return (lat, lon) center of a 'minlat,minlon,maxlat,maxlon' bbox string."""
    minlat, minlon, maxlat, maxlon = map(float, bbox_str.split(","))
    return (minlat + maxlat) / 2, (minlon + maxlon) / 2


def _point_in_bbox(lat, lon, bbox_str):
    minlat, minlon, maxlat, maxlon = map(float, bbox_str.split(","))
    return minlat <= lat <= maxlat and minlon <= lon <= maxlon


def _feature_centroid(feature):
    """Rough centroid from the first ring of coordinates."""
    try:
        geom = feature["geometry"]
        coords = geom["coordinates"]
        if geom["type"] == "Polygon":
            ring = coords[0]
        elif geom["type"] == "MultiPolygon":
            ring = coords[0][0]
        else:
            return None, None
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        return sum(lats) / len(lats), sum(lons) / len(lons)
    except Exception:
        return None, None


def filter_geojson_by_region(gj, region_name):
    """Return a filtered GeoJSON keeping only features that fall within the
    bounding box of *region_name* (looked up in RA_BBOX).
    Returns (filtered_gj, bbox_str_or_None)."""
    region_n = _norm(region_name)
    bbox_str = None
    for key, val in RA_BBOX.items():
        if region_n in key or key in region_n:
            bbox_str = val
            break
    if bbox_str is None:
        return gj, None  # unknown region → return everything

    filtered = [
        feat for feat in gj["features"]
        if _point_in_bbox(*_feature_centroid(feat), bbox_str)
    ]
    filtered_gj = {"type": "FeatureCollection", "features": filtered}
    return filtered_gj, bbox_str


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
def load_ra_geojson():
    """Load Administrative Regions (Regiões Administrativas) from rasDF.json."""
    path = os.path.join(os.path.dirname(__file__), "database", "rasDF.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    df = pd.read_csv(path, sep=";")          # <- adicione sep=";"
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
    return (
        df.groupby("i_desc_estab_cnes_notif")
        .size()
        .reset_index(name="casos")
        .rename(columns={"i_desc_estab_cnes_notif": "estab"})
    )


@st.cache_data(show_spinner=False)
def attach_dengue_to_sectors(ra_name):
    """Attach dengue case counts to individual census sectors of a single RA,
    matched by NM_SUBDIST (which equals the RA name in setoresDF.json)."""
    import copy
    gj, by_code, by_subdist = load_geojson()
    df  = load_dengue_data()
    agg = aggregate_dengue_by_sector(df)
    ubs = load_ubs_df()
    agg["estab_norm"] = agg["estab"].apply(_norm)
    ubs["nome_norm2"] = ubs["nome"].apply(_norm)
    estab_map = dict(zip(agg["estab_norm"], agg["casos"]))

    # NM_SUBDIST == RA name — direct match (deep-copy to avoid mutating cache)
    ra_n = _norm(ra_name)
    ra_features = [
        copy.deepcopy(feat)
        for key, feats in by_subdist.items() for feat in feats
        if ra_n in _norm(key) or _norm(key) in ra_n
    ]

    for feat in ra_features:
        feat["properties"]["dengue_casos"] = 0

    for _, row in ubs.iterrows():
        bairro_n = _norm(row.get("bairro", ""))
        if ra_n not in bairro_n and bairro_n not in ra_n:
            continue
        nome_n = row["nome_norm2"]
        casos  = estab_map.get(nome_n, 0)
        if casos == 0:
            for estab_n, c in estab_map.items():
                if nome_n in estab_n or estab_n in nome_n:
                    casos = c
                    break
        for feat in ra_features:
            if _norm(feat["properties"].get("NM_SUBDIST", "")) == bairro_n:
                feat["properties"]["dengue_casos"] += casos
                break

    return {"type": "FeatureCollection", "features": ra_features}


@st.cache_data(show_spinner=False)
def attach_dengue_to_ra():
    """Attach dengue case counts to Administrative Region polygons (rasDF.json)."""
    ra_gj = load_ra_geojson()
    df    = load_dengue_data()
    agg   = aggregate_dengue_by_sector(df)
    ubs   = load_ubs_df()
    agg["estab_norm"] = agg["estab"].apply(_norm)
    ubs["nome_norm2"] = ubs["nome"].apply(_norm)
    estab_map = dict(zip(agg["estab_norm"], agg["casos"]))

    # Map UBS bairro → caso count
    bairro_casos: dict = {}
    for _, row in ubs.iterrows():
        bairro_n = _norm(row.get("bairro", ""))
        nome_n   = row["nome_norm2"]
        casos    = estab_map.get(nome_n, 0)
        if casos == 0:
            for estab_n, c in estab_map.items():
                if nome_n in estab_n or estab_n in nome_n:
                    casos = c
                    break
        bairro_casos[bairro_n] = bairro_casos.get(bairro_n, 0) + casos

    # Aggregate bairro counts into RAs using RA_BBOX name matching
    import copy
    ra_gj_copy = copy.deepcopy(ra_gj)
    for feat in ra_gj_copy["features"]:
        ra_n  = _norm(feat["properties"]["ra"])
        total = sum(
            casos for bairro_n, casos in bairro_casos.items()
            if ra_n in bairro_n or bairro_n in ra_n
        )
        feat["properties"]["dengue_casos"] = total

    return ra_gj_copy


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

@st.cache_data(show_spinner=False)
def get_dengue_summary(region: str = None):
    df = load_dengue_data()

    # --- regional filter ------------------------------------------------
    region_label = None
    if region:
        region_n = _norm(region)
        ubs = load_ubs_df()
        ubs["nome_norm2"] = ubs["nome"].apply(_norm)

        # find UBS rows whose bairro or nome match the requested region
        matching_ubs = ubs[
            ubs["bairro_norm"].str.contains(region_n, na=False) |
            ubs["nome_norm"].str.contains(region_n, na=False)
        ]

        if not matching_ubs.empty:
            # build a set of normalised establishment names for the region
            estab_norms = set(matching_ubs["nome_norm2"].tolist())
            df["estab_norm_tmp"] = df["i_desc_estab_cnes_notif"].apply(_norm)
            df = df[
                df["estab_norm_tmp"].apply(
                    lambda x: any(e in x or x in e for e in estab_norms)
                )
            ]
            region_label = region.title()
        else:
            # fallback: text-search within the establishment name column
            df = df[
                df["i_desc_estab_cnes_notif"].apply(_norm).str.contains(region_n, na=False)
            ]
            region_label = region.title() if not df.empty else None
    # --------------------------------------------------------------------

    total = len(df)
    confirmed = len(df[df["i_desc_classificacao"].str.contains("Dengue", na=False)])
    alarm = len(df[df["i_desc_classificacao"] == "Dengue com sinais de alarme"])
    deaths = len(df[df["i_desc_evolucao"] == "Óbito pelo agravo notificado"])

    by_estab = (
        df.groupby("i_desc_estab_cnes_notif")
        .size()
        .reset_index(name="casos")
        .sort_values("casos", ascending=False)
    )

    ubs = load_ubs_df()
    ubs["nome_norm2"] = ubs["nome"].apply(_norm)
    df["estab_norm"] = df["i_desc_estab_cnes_notif"].apply(_norm)

    estab_bairro = {}
    for _, row in ubs.iterrows():
        estab_bairro[row["nome_norm2"]] = row.get("bairro", "Não Informado")

    df["bairro"] = df["estab_norm"].map(
        lambda x: next((estab_bairro[k] for k in estab_bairro if k in x or x in k), "Não Informado")
    )

    by_bairro = (
        df[df["bairro"] != "Não Informado"]
        .groupby("bairro")
        .size()
        .reset_index(name="casos")
        .sort_values("casos", ascending=False)
        .head(5)
    )

    by_age = (
        df.groupby("i_faixa_etaria")
        .size()
        .reset_index(name="casos")
        .sort_values("casos", ascending=False)
        .head(3)
    )

    by_week = (
        df.groupby("i_ano_semana_prim_sintomas_svs")
        .size()
        .reset_index(name="casos")
        .sort_values("i_ano_semana_prim_sintomas_svs")
    )
    peak_week = by_week.loc[by_week["casos"].idxmax(), "i_ano_semana_prim_sintomas_svs"]

    return {
        "total": total,
        "confirmed": confirmed,
        "alarm": alarm,
        "deaths": deaths,
        "top_estab": by_estab.head(5).to_dict("records"),
        "top_bairro": by_bairro.to_dict("records"),
        "top_age": by_age.to_dict("records"),
        "peak_week": peak_week,
        "by_week": by_week.to_dict("records"),
        "region_label": region_label,
    }

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


@st.cache_data(show_spinner=False, ttl=3600)
def translate_to_pt(text):
    """Translate any text to Brazilian Portuguese for corpus retrieval.
    Returns the original text unchanged if already in Portuguese."""
    resp = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the user message to Brazilian Portuguese. "
                    "If it is already in Portuguese, return it unchanged. "
                    "Return ONLY the translated text, no explanation, no markdown."
                ),
            },
            {"role": "user", "content": f"/no_think {text}"},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    result = resp.choices[0].message.content.strip()
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


def answer_health_question(pergunta, lang="en"):
    dense_retriever, bm25, chunks = setup_retriever()
    entities = extract_entities(pergunta)

    # Translate to Portuguese for retrieval against the Portuguese corpus
    pergunta_pt = translate_to_pt(pergunta) if lang == "en" else pergunta

    dense_docs = dense_retriever.invoke(pergunta_pt)

    query_terms = pergunta_pt.lower().split()
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
    return raw, localidades_encontradas, entities

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
    lang = re.sub(r"<think>.*?</think>", "", lang, flags=re.DOTALL).strip()
    return "pt" if "pt" in lang else "en"


def parse_command(user_text):
    ra_gj        = load_ra_geojson()
    ra_names_str = ", ".join(sorted(f["properties"]["ra"] for f in ra_gj["features"]))
    poi_cats_str = ", ".join(sorted({_norm(k) for k in OSM_CATEGORIES}))

    system = f"""You are the map control agent for the Opossum app (Fiocruz Brasília, DF, Brazil).
Classify the user message and respond ONLY with pure JSON (no markdown, no explanation).

Actions:
- "draw":     highlight an Administrative Region (RA) boundary polygon on the map.
              Use when the user asks to "show", "draw", "highlight", or "zoom to" an RA.
- "setor":    show individual census sectors (setores censitários) inside an RA.
              ONLY triggered when the user explicitly mentions "setor censitário",
              "census sector", "setores", "census sectors", or asks to see sectors.
              "target" = RA name, or null to show all sectors in the DF.
- "remove":   remove a specific layer from the map
- "clear":    remove ALL layers from the map
- "poi":      search for points of interest on OpenStreetMap (hospitals, pharmacies, clinics, etc.)
              ONLY when the user wants to FIND or LOCATE facilities on the map
              (e.g. "show hospitals in Ceilândia", "where are the pharmacies near me").
              NOT for questions about how a service works, its coverage, schedules, or procedures.
- "geocode":  locate and pin a specific address or place on the map
- "dengue":       display a dengue case choropleth map.
                  Triggered by: "show dengue map", "dengue map", "mapa dengue",
                  "show me dengue", "visualize dengue", and similar.
                  If the user mentions a specific RA name (from the list above), set "target" to that RA name.
                  If the user mentions only "Brasília", "DF", "Distrito Federal", or the whole city/country, set "target" to null.
- "dengue_query": answer analytical questions about dengue data.
                  Triggered by: "how many cases", "which region has more",
                  "worst region", "most affected", "dengue situation",
                  "how is dengue", "dengue cases in 2026", "peak week",
                  "most cases", "qual região", "quantos casos", "região mais afetada"
                  If the user mentions a specific region/neighbourhood, set "target" to that region name.
- "risco":    display geological risk areas on the map (CPRM data).
              Triggered by: "risk zones", "geological risk", "landslide", "risco geológico",
              "deslizamento", "áreas de risco", "risk areas", "enxurrada", "voçoroca",
              "risco", "risk", "are there risks", "zonas de risco", "near [location] risk"
              If the user mentions a specific region/neighbourhood, set "area" to that region name.
- "queimada": display burned areas on the map (2025 data).
              Triggered by: "burned areas", "fire", "queimadas", "incêndio", "áreas queimadas",
              "wildfires", "queimou", "fogo"
              If the user mentions a specific region/neighbourhood, set "area" to that region name.
- "none":     ANY question about public health topics, services, policies, or information —
              including questions about how a unit works, its coverage area, opening hours,
              referral procedures, target populations, or specific health programmes.
              When in doubt between "poi" and "none", choose "none".

Available RAs (Portuguese names): {ra_names_str}
Available POI categories: {poi_cats_str}

Response format:
{{
  "action":      "draw"|"setor"|"remove"|"clear"|"poi"|"geocode"|"dengue"|"dengue_query"|"risco"|"queimada"|"none",
  "target":      "<RA name, sector code, full address or category, or null>",
  "area":        "<RA/neighbourhood name for POI/risco/queimada search, or null>",
  "category":    "<normalised POI category without accents, or null>",
  "name_filter": "<specific name/number to filter, e.g. '01', 'Asa Norte', or null>"
}}

Rules:
- "draw" shows RA-level boundaries; "setor" reveals the finer census-sector grid — only on explicit user request.
- "poi" requires an explicit intent to LOCATE something on the map. Questions that merely mention
  a health facility by name (e.g. "Como funciona o Cerest?", "Qual a cobertura do CAPS?",
  "O CRE Oeste atende Brazlândia?") are ALWAYS "none", never "poi".
- Questions about geological risks, landslides, floods, burned areas, or fire are ALWAYS
  "risco" or "queimada" actions, NEVER "none".
- "near [location]" + risk/fire topic = "risco"/"queimada" with area set to that location.
- For "poi": if the user mentions a specific unit name or number (e.g. "UBS 01"), set "name_filter".
- For "geocode": "target" = full address or place name.
- For "draw"/"setor"/"remove": "target" = RA name (normalised to the list above).
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


def has_map_intent(text):
    """Return True if the query explicitly requests map drawing or implicitly
    implies spatial lookup (proximity, listing, location of facilities)."""
    t = _norm(text)

    # Explicit drawing / visualisation verbs
    explicit = [
        "desenh", "mostr", "renderiz", "exib", "destac", "plot", "mapeie", "mapa",
        "draw", "show", "render", "display", "highlight", "map", "visuali",
    ]
    # Implicit spatial-lookup patterns
    implicit = [
        "próxim", "perto", "mais perto", "mais próxim", "próximo a", "perto de",
        "near", "nearby", "closest", "nearest", "around",
        "quais .{0,30} exist", "onde fica", "onde estão", "onde há", "onde tem",
        "where is", "where are", "where can i find",
        "lista .{0,20} em", "list .{0,20} in",
    ]

    if any(kw in t for kw in explicit):
        return True
    if any(re.search(pat, t) for pat in implicit):
        return True
    return False


def match_facilities_from_entities(entities):
    """Given NER entities, search the local UBS database for matching facilities.
    Returns a list of POI dicts (name, lat, lon, address, phone) or empty list."""
    ubs = load_ubs_df()
    terms = [
        _norm(e)
        for e in entities.get("organizations", []) + entities.get("locations", [])
        if e
    ]
    if not terms:
        return []

    matched = set()
    pois = []
    for term in terms:
        hits = ubs[
            ubs["nome_norm"].str.contains(term, na=False) |
            ubs["bairro_norm"].str.contains(term, na=False)
        ]
        for _, row in hits.iterrows():
            key = row["nome_norm"]
            if key not in matched:
                matched.add(key)
                pois.append({
                    "name":    row["nome"].title(),
                    "lat":     row["latitude"],
                    "lon":     row["longitude"],
                    "address": row.get("logradouro", ""),
                    "phone":   "",
                })
    return pois


def execute_command(parsed, lang="en"):
    action   = parsed.get("action",   "none")
    target   = parsed.get("target",   "") or ""
    area     = parsed.get("area",     None)
    category = parsed.get("category", None) or target

    def msg(en, pt):
        return pt if lang == "pt" else en

    if action == "clear":
        for store in ["drawn_layers", "ra_layers", "poi_layers", "pin_layers"]:
            st.session_state[store] = {}
        st.session_state.pop("dengue_layer",  None)
        st.session_state.pop("risco_layer",   None)
        st.session_state.pop("queimada_layer", None)
        return msg("🗺️ All layers have been removed from the map.",
                   "🗺️ Todas as camadas foram removidas do mapa.")

    if action == "remove":
        key_norm = _norm(target)
        removed  = []
        for store in ["drawn_layers", "ra_layers", "poi_layers", "pin_layers"]:
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
        # Highlight the RA boundary polygon
        ra_gj = load_ra_geojson()
        ra_norm = _norm(target)
        matched = [
            f for f in ra_gj["features"]
            if ra_norm in _norm(f["properties"]["ra"])
            or _norm(f["properties"]["ra"]) in ra_norm
        ]
        if not matched:
            return msg(
                f"⚠️ No Administrative Region found for **{target}**.\n"
                "Try a name from the list (e.g. Ceilândia, Taguatinga, Asa Norte).",
                f"⚠️ Nenhuma Região Administrativa encontrada para **{target}**.\n"
                "Tente um nome da lista (ex: Ceilândia, Taguatinga, Asa Norte).",
            )
        label = matched[0]["properties"]["ra"]
        color = next_poly_color()
        st.session_state["ra_layers"][label] = {"features": matched, "color": color}
        # Centre map on RA bbox
        all_lats, all_lons = [], []
        for feat in matched:
            geom = feat["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                for ring in poly:
                    for lon, lat in ring:
                        all_lats.append(lat); all_lons.append(lon)
        if all_lats:
            st.session_state["map_center"] = [
                sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
        return msg(
            f"✅ **{label}** highlighted on the map.\n"
            f"_To see individual census sectors, type 'show census sectors of {label}'._",
            f"✅ **{label}** destacada no mapa.\n"
            f"_Para ver os setores censitários individuais, digite 'mostrar setores censitários de {label}'._",
        )

    if action == "setor":
        # Show individual census sectors — only on explicit user request
        _, by_code, by_subdist = load_geojson()
        if target:
            label, features = find_sectors(target)
            if not features:
                return msg(
                    f"⚠️ No census sectors found for **{target}**.\n"
                    "Try the name of an Administrative Region (e.g. Ceilândia, Taguatinga).",
                    f"⚠️ Nenhum setor censitário encontrado para **{target}**.\n"
                    "Tente o nome de uma RA (ex: Ceilândia, Taguatinga).",
                )
        else:
            label = "DF (all sectors)"
            features = [f for feats in by_subdist.values() for f in feats]
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
        return msg(
            f"✅ **{len(features)} census sector(s)** for **{label}** drawn on the map.",
            f"✅ **{len(features)} setor(es) censitário(s)** de **{label}** desenhados no mapa.",
        )

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
        region = target or area or None

        # "Brasília", "Brasilia", "DF", "Distrito Federal" all mean the whole DF
        DF_ALIASES = {"brasilia", "brasília", "df", "distrito federal", "federal district"}
        if region and _norm(region) in DF_ALIASES:
            region = None

        df = load_dengue_data()
        total     = len(df)
        confirmed = len(df[df["i_desc_classificacao"].str.contains("Dengue", na=False)])
        alarm     = len(df[df["i_desc_classificacao"] == "Dengue com sinais de alarme"])
        deaths    = len(df[df["i_desc_evolucao"] == "Óbito pelo agravo notificado"])

        if region:
            # Sector-level choropleth for a single RA
            gj = attach_dengue_to_sectors(region)
            st.session_state["dengue_layer"] = gj
            st.session_state["dengue_mode"]  = "sector"
            # Centre on RA
            bbox_str = None
            for key, val in RA_BBOX.items():
                if _norm(region) in key or key in _norm(region):
                    bbox_str = val
                    break
            if bbox_str:
                center_lat, center_lon = _bbox_center(bbox_str)
                st.session_state["map_center"] = [center_lat, center_lon]
            scope_en = f" in **{region.title()}**"
            scope_pt = f" em **{region.title()}**"
        else:
            # RA-level choropleth for the whole DF
            gj = attach_dengue_to_ra()
            st.session_state["dengue_layer"] = gj
            st.session_state["dengue_mode"]  = "ra"
            st.session_state["map_center"]   = [-15.793889, -47.882778]
            scope_en = " for the **Distrito Federal**"
            scope_pt = " para o **Distrito Federal**"

        return msg(
            f"🦟 Dengue map loaded{scope_en} — 2026. "
            f"**{total:,}** notified cases, **{confirmed}** confirmed "
            f"({alarm} with warning signs), **{deaths}** death(s).",
            f"🦟 Mapa de dengue carregado{scope_pt} — 2026. "
            f"**{total:,}** casos notificados, **{confirmed}** confirmados "
            f"({alarm} com sinais de alarme), **{deaths}** óbito(s).",
        )

    if action == "risco":
        gj_full = load_risco_geologico()
        region = target or area or None
        if region and _norm(region) in {"brasilia", "brasília", "df", "distrito federal", "federal district"}:
            region = None
        if region:
            gj, bbox_str = filter_geojson_by_region(gj_full, region)
            if bbox_str:
                center_lat, center_lon = _bbox_center(bbox_str)
                st.session_state["map_center"] = [center_lat, center_lon]
            else:
                st.session_state["map_center"] = [-15.793889, -47.882778]
        else:
            gj = gj_full
            st.session_state["map_center"] = [-15.793889, -47.882778]
        st.session_state["risco_layer"] = gj
        n = len(gj["features"])
        scope_en = f" in **{region.title()}**" if region else ""
        scope_pt = f" em **{region.title()}**" if region else ""
        if n == 0 and region:
            return msg(
                f"⚠️ No geological risk sectors found for **{region.title()}**.",
                f"⚠️ Nenhum setor de risco geológico encontrado para **{region.title()}**.",
            )
        return msg(f"⚠️ {n} geological risk sector(s) loaded{scope_en} (Alto and Muito Alto).",
                   f"⚠️ {n} setor(es) de risco geológico carregado(s){scope_pt} (Alto e Muito Alto).")

    if action == "queimada":
        gj_full = load_queimadas()
        region = target or area or None
        if region and _norm(region) in {"brasilia", "brasília", "df", "distrito federal", "federal district"}:
            region = None
        if region:
            gj, bbox_str = filter_geojson_by_region(gj_full, region)
            if bbox_str:
                center_lat, center_lon = _bbox_center(bbox_str)
                st.session_state["map_center"] = [center_lat, center_lon]
            else:
                st.session_state["map_center"] = [-15.793889, -47.882778]
        else:
            gj = gj_full
            st.session_state["map_center"] = [-15.793889, -47.882778]
        st.session_state["queimada_layer"] = gj
        n = len(gj["features"])
        scope_en = f" in **{region.title()}**" if region else ""
        scope_pt = f" em **{region.title()}**" if region else ""
        if n == 0 and region:
            return msg(
                f"⚠️ No burned area polygons found for **{region.title()}** in 2025.",
                f"⚠️ Nenhum polígono de área queimada encontrado para **{region.title()}** em 2025.",
            )
        return msg(f"🔥 {n} burned area polygon(s) loaded{scope_en} (2025 data).",
                   f"🔥 {n} polígono(s) de áreas queimadas carregado(s){scope_pt} (dados 2025).")

    if action == "dengue_query":
        region = target or area or None
        if region and _norm(region) in {"brasilia", "brasília", "df", "distrito federal", "federal district"}:
            region = None
        s = get_dengue_summary(region=region)

        scope_en = f"in **{s['region_label']}**" if s.get("region_label") else "in the **Distrito Federal**"
        scope_pt = f"em **{s['region_label']}**" if s.get("region_label") else "no **Distrito Federal**"

        no_data_en = f"⚠️ No dengue records found for **{region}** in 2026. The region may not appear as a notifying area in the dataset."
        no_data_pt = f"⚠️ Nenhum registro de dengue encontrado para **{region}** em 2026. A região pode não constar como área notificadora no conjunto de dados."

        if s["total"] == 0 and region:
            return msg(no_data_en, no_data_pt)

        top_estab_txt = "\n".join(
            f"  {r['i_desc_estab_cnes_notif']}: {r['casos']} cases"
            for r in s["top_estab"]
        )
        top_bairro_txt = "\n".join(
            f"  {r['bairro'].title()}: {r['casos']} cases"
            for r in s["top_bairro"]
        ) if s["top_bairro"] else "  (geographic breakdown not available)"

        top_age_txt = "\n".join(
            f"  {r['i_faixa_etaria'].replace('_', '-')}: {r['casos']} cases"
            for r in s["top_age"]
        )

        return msg(
            f"🦟 **Dengue {scope_en} — 2026**\n\n"
            f"**Total notified cases:** {s['total']:,}\n"
            f"**Confirmed dengue:** {s['confirmed']} ({s['alarm']} with warning signs)\n"
            f"**Deaths:** {s['deaths']}\n"
            f"**Epidemiological peak:** week {str(s['peak_week'])[4:]} of 2026\n\n"
            f"**Top notifying facilities:**\n{top_estab_txt}\n\n"
            f"**Most affected neighborhoods:**\n{top_bairro_txt}\n\n"
            f"**Most affected age groups:**\n{top_age_txt}\n\n"
            f"_Type 'show dengue map' to visualize case distribution by census sector._",
            f"🦟 **Dengue {scope_pt} — 2026**\n\n"
            f"**Total de casos notificados:** {s['total']:,}\n"
            f"**Dengue confirmada:** {s['confirmed']} ({s['alarm']} com sinais de alarme)\n"
            f"**Óbitos:** {s['deaths']}\n"
            f"**Pico epidemiológico:** semana {str(s['peak_week'])[4:]} de 2026\n\n"
            f"**Estabelecimentos com mais notificações:**\n{top_estab_txt}\n\n"
            f"**Bairros mais afetados:**\n{top_bairro_txt}\n\n"
            f"**Faixas etárias mais afetadas:**\n{top_age_txt}\n\n"
            f"_Digite 'mostrar mapa de dengue' para visualizar a distribuição por setor censitário._",
        )

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

if "drawn_layers"  not in st.session_state: st.session_state["drawn_layers"]  = {}
if "ra_layers"     not in st.session_state: st.session_state["ra_layers"]     = {}
if "poi_layers"    not in st.session_state: st.session_state["poi_layers"]    = {}
if "pin_layers"    not in st.session_state: st.session_state["pin_layers"]    = {}
if "map_center"    not in st.session_state: st.session_state["map_center"]    = [-15.793889, -47.882778]
if "chat_history"  not in st.session_state: st.session_state["chat_history"]  = []
if "dengue_mode"   not in st.session_state: st.session_state["dengue_mode"]   = "ra"

col_chat, col_map = st.columns([1, 1])

with col_chat:
    st.subheader("Assistant")

    for msg_item in st.session_state["chat_history"]:
        role_label = "You" if msg_item["role"] == "user" else "Agent"
        with st.chat_message(msg_item["role"]):
            st.markdown(f"**{role_label}:** {msg_item['content']}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type a command or question:",
            placeholder="e.g. Hospitals in Ceilândia | Burned areas | Geological risk | Dengue map",
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
                response, localidades, entities = answer_health_question(user_msg, lang=lang)

                if has_map_intent(user_msg):
                    # ── Pin facilities found in the local UBS database ────────
                    facility_pois = match_facilities_from_entities(entities)
                    if facility_pois:
                        layer_label = "🏥 Unidades mencionadas"
                        st.session_state["poi_layers"][layer_label] = {
                            "pois": facility_pois, "icon": "🏥", "color": "blue"
                        }
                        lats = [p["lat"] for p in facility_pois]
                        lons = [p["lon"] for p in facility_pois]
                        st.session_state["map_center"] = [
                            sum(lats) / len(lats), sum(lons) / len(lons)
                        ]

                    # ── Highlight RAs mentioned in retrieved documents ─────────
                    entidades_loc = set(entities["locations"])
                    if entidades_loc:
                        localidades = {
                            loc for loc in localidades
                            if any(e in _norm(loc) or _norm(loc) in e for e in entidades_loc)
                        }
                    last_lats, last_lons = [], []
                    for loc in localidades:
                        ra_gj = load_ra_geojson()
                        loc_n = _norm(loc)
                        matched_ra = [
                            f for f in ra_gj["features"]
                            if loc_n in _norm(f["properties"]["ra"])
                            or _norm(f["properties"]["ra"]) in loc_n
                        ]
                        if matched_ra:
                            ra_label = matched_ra[0]["properties"]["ra"]
                            color = next_poly_color()
                            st.session_state["ra_layers"][ra_label] = {
                                "features": matched_ra, "color": color,
                            }
                            geom = matched_ra[0]["geometry"]
                            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
                            for poly in polys:
                                for ring in poly:
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

with col_map:
    st.subheader("Map")

    all_labels = (
        list(st.session_state["ra_layers"].keys())
        + list(st.session_state["drawn_layers"].keys())
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
            for store in ["drawn_layers", "ra_layers", "poi_layers", "pin_layers"]:
                st.session_state[store] = {}
            for k in ["dengue_layer", "risco_layer", "queimada_layer"]:
                st.session_state.pop(k, None)
            st.rerun()

    center = st.session_state["map_center"]
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="OpenStreetMap", control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Relief", control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Satellite", control=True,
    ).add_to(m)

    if "dengue_layer" in st.session_state:
        dengue_mode = st.session_state.get("dengue_mode", "ra")
        dengue_feats = st.session_state["dengue_layer"].get("features", [])
        if dengue_feats:
            if dengue_mode == "ra":
                tooltip_fields  = ["ra", "dengue_casos"]
                tooltip_aliases = ["Region:", "Cases:"]
            else:
                tooltip_fields  = ["NM_SUBDIST", "dengue_casos"]
                tooltip_aliases = ["Subregions:", "Cases:"]
            folium.GeoJson(
                st.session_state["dengue_layer"],
                name="Dengue 2026",
                style_function=lambda feature: {
                    "fillColor": get_dengue_color(feature["properties"]["dengue_casos"]),
                    "color": "black", "weight": 0.3, "fillOpacity": 0.7,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                ),
            ).add_to(m)

    if "risco_layer" in st.session_state:
        folium.GeoJson(
            st.session_state["risco_layer"],
            name="Geological Risk (CPRM)",
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

    if "queimada_layer" in st.session_state:
        folium.GeoJson(
            st.session_state["queimada_layer"],
            name="Fires 2025",
            style_function=lambda feat: {
                "fillColor":   QUEIMADA_COLORS.get(feat["properties"]["mes"], "#fdae61"),
                "color": "#333", "weight": 0.5, "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["mes_nome", "area_ha", "data"],
                aliases=["Month:", "Area:", "Date:"],
            ),
        ).add_to(m)

    # ── Highlighted RA polygons (from "draw" command) ──────────────────────
    for label, layer in st.session_state["ra_layers"].items():
        color   = layer["color"]
        payload = {"type": "FeatureCollection", "features": layer["features"]}
        folium.GeoJson(
            payload, name=f"RA: {label}",
            style_function=lambda _, c=color: {
                "fillColor": c, "color": c, "weight": 2.5, "fillOpacity": 0.25,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["legenda"],
                aliases=[""],
                style="font-size: 13px; font-weight: bold;",
            ),
        ).add_to(m)

    # ── Census sectors (from explicit "setor" command only) ────────────────
    for label, layer in st.session_state["drawn_layers"].items():
        color   = layer["color"]
        payload = {"type": "FeatureCollection", "features": layer["features"]}
        folium.GeoJson(
            payload, name=f"Setores: {label}",
            style_function=lambda _, c=color: {
                "fillColor": c, "color": c, "weight": 0.8, "fillOpacity": 0.30,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["CD_SETOR", "NM_SUBDIST"],
                aliases=["Sector:", "Region:"],
            ),
        ).add_to(m)

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

    for label, pin in st.session_state["pin_layers"].items():
        folium.Marker(
            location=[pin["lat"], pin["lon"]],
            popup=folium.Popup(pin["display_name"], max_width=300),
            tooltip=label,
            icon=folium.Icon(color="darkred", icon="map-marker"),
        ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    st_folium(m, width=None, height=500, returned_objects=[])


st.markdown("---")
#st.markdown("© 2026 · Opossum – Fiocruz Brasília · Grupo de Inteligência Computacional na Saúde (GICS)")
