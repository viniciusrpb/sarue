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

st.set_page_config(page_title="Saruê ::: Fiocruz Brasília", layout="wide")
st.title("Saruê ::: Fiocruz Brasília")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

HEADERS_OSM = {"User-Agent": "Sarue-Fiocruz/2.0 (fiocruz.br)"}

OSM_CATEGORIES = {
    "hospital": ("amenity", "hospital", "🏥", "red"),
    "ubs": ("amenity", "clinic", "🏥", "blue"),
    "ups": ("amenity", "clinic", "🏥", "blue"),
    "clinica": ("amenity", "clinic", "🏥", "blue"),
    "clínica": ("amenity", "clinic", "🏥", "blue"),
    "farmacia": ("amenity", "pharmacy", "💊", "green"),
    "farmácia": ("amenity", "pharmacy", "💊", "green"),
    "medico": ("amenity", "doctors", "👨‍⚕️", "cadetblue"),
    "médico": ("amenity", "doctors", "👨‍⚕️", "cadetblue"),
    "dentista": ("amenity", "dentist", "🦷", "purple"),
    "social": ("amenity", "social_facility", "🤝", "orange"),
    "cras": ("amenity", "social_facility", "🤝", "orange"),
    "creas": ("amenity", "social_facility", "🤝", "orange"),
    "escola": ("amenity", "school", "🏫", "darkblue"),
    "creche": ("amenity", "kindergarten", "👶", "pink"),
    "saude": ("healthcare", "*", "➕", "red"),
    "saúde": ("healthcare", "*","➕", "red"),
    "academia": ("leisure", "fitness_centre", "🏋️", "darkgreen"),
}

DF_BBOX = "-16.05,-48.28,-15.48,-47.30"

def _norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

@st.cache_data(show_spinner=False)
def load_geojson():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "database", "setoresDF.json")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)

    by_code    = {}
    by_subdist = {}
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
    base_dir  = os.path.join(os.path.dirname(__file__), "samples")
    documents = []
    for path in ["noticias_ses_1_100.csv"]:
        df = pd.read_csv(
            os.path.join(base_dir, path),
            sep=",", engine="python",
            on_bad_lines="skip", encoding="utf-8",
        )
        for _, row in df.iterrows():
            titulo  = str(row.get("title",   "")).strip()
            noticia = str(row.get("content", "")).strip()
            text = f"{titulo}\n\n{noticia}".strip()
            if len(text) > 50:
                documents.append( Document(page_content=text, metadata={"source": path}) )
    return documents

@st.cache_data
def load_dengue_data():
    path = os.path.join(os.path.dirname(__file__), "database", "dados_dengue-16042026-ano_2026.csv")
    df = pd.read_csv(path)

    df.columns = [c.lower() for c in df.columns]

    return df

def aggregate_dengue_by_sector(df):
    # exemplo: coluna 'cd_setor' e 'casos'
    grouped = df.groupby("cd_setor")["casos"].sum().reset_index()
    return grouped

def attach_dengue_to_geojson():
    gj, by_code, _ = load_geojson()
    df = load_dengue_data()
    agg = aggregate_dengue_by_sector(df)

    dengue_map = dict(zip(agg["cd_setor"], agg["casos"]))

    for feat in gj["features"]:
        code = feat["properties"].get("CD_SETOR")
        feat["properties"]["dengue_casos"] = dengue_map.get(code, 0)

    return gj

@st.cache_resource(show_spinner=False)
def setup_retriever():
    docs = load_documents()
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="juridics/bertimbau-base-portuguese-sts-scale") #vou tentar isso aqui, eh novo!!!

    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

def geocode(address):

    params = {
        "q": f"{address}, Distrito Federal, Brasil",
        "format": "json",
        "limit": 5,
        "addressdetails": 1,
        "countrycodes": "br",
        "viewbox": "-48.28,-16.05,-47.30,-15.48",
        "bounded": 1,
    }
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=HEADERS_OSM,
            timeout=10,
        )

        r.raise_for_status()
        results = r.json()
        return [
            {
                "display_name": d["display_name"],
                "lat": float(d["lat"]),
                "lon": float(d["lon"]),
                "osm_type": d.get("type", ""),

            }
            for d in results
        ], None
    except requests.exceptions.Timeout:
        return [], "Timeout ao contatar Nominatim. Tente novamente."
    except Exception as e:
        return [], str(e)

def _overpass_query(key, value, area_name):

    if area_name:
        area_filter = f'area["name"~"{area_name}",i]["admin_level"~"8|9|10"]->.a;'
        if value == "*":
            selectors = f'node["{key}"](area.a); way["{key}"](area.a);'
        else:
            selectors = f'node["{key}"="{value}"](area.a); way["{key}"="{value}"](area.a);'
        return f"[out:json][timeout:25];\n{area_filter}\n({selectors});\nout center 100;"
    else:
        if value == "*":
            selectors = f'node["{key}"]({DF_BBOX}); way["{key}"]({DF_BBOX});'
        else:
            selectors = f'node["{key}"="{value}"]({DF_BBOX}); way["{key}"="{value}"]({DF_BBOX});'
        return f"[out:json][timeout:25];\n({selectors});\nout center 100;"


def search_poi(category, area_name):

    cat_norm = _norm(category)

    matched = None
    for k, v in OSM_CATEGORIES.items():
        if cat_norm == _norm(k) or _norm(k) in cat_norm or cat_norm in _norm(k):
            matched = v
            break

    if matched:
        osm_key, osm_val, icon, color = matched
    else:
        osm_key, osm_val, icon, color = "amenity", cat_norm, "📍", "gray"

    query = _overpass_query(osm_key, osm_val, area_name)

    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers=HEADERS_OSM,
            timeout=30,
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except requests.exceptions.Timeout:
        return [], icon, color, "Timeout ao consultar a Overpass API."
    except Exception as e:
        return [], icon, color, str(e)

    pois = []
    for el in elements:
        tags = el.get("tags", {})
        name = (
            tags.get("name")
            or tags.get("name:pt")
            or tags.get("operator")
            or osm_val.title()
        )
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")

        if lat and lon:
            pois.append({
                "name":    name,
                "lat":     lat,
                "lon":     lon,
                "address": tags.get("addr:street", ""),
                "phone":   tags.get("phone", tags.get("contact:phone", "")),
            })

    return pois, icon, color, None

def next_poly_color():
    return POLY_COLORS[len(st.session_state["drawn_layers"]) % len(POLY_COLORS)]

def find_sectors(query_text: str):
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

def parse_command(user_text: str) -> dict:
    subdist_str  = ", ".join(get_subdist_list())
    poi_cats_str = ", ".join(sorted({_norm(k) for k in OSM_CATEGORIES}))

    system = f"""Você é o agente de controle de mapa do app Saruê (Fiocruz Brasília, DF).
Classifique a mensagem e responda APENAS com JSON puro (sem markdown, sem explicação).

Ações:
- "draw": desenhar polígonos de setor censitário de uma RA no mapa
- "remove": remover uma camada específica do mapa
- "clear": remover TODAS as camadas do mapa
- "poi": buscar pontos de interesse no OpenStreetMap (hospitais, farmácias, UBS etc.)
- "geocode": localizar e marcar um endereço ou lugar específico no mapa
- "none": pergunta de saúde pública ou assunto não relacionado ao mapa
- "dengue": visualizar casos de dengue no mapa (mapa temático por setor)

RAs disponíveis (para draw/remove): {subdist_str}
Categorias de POI disponíveis: {poi_cats_str}

Formato de resposta:
{{
  "action":   "draw"|"remove"|"clear"|"poi"|"geocode"|"none",
  "target":   "<RA, código setor, endereço completo ou categoria>",
  "area":     "<nome da RA onde buscar POIs, ou null>",
  "category": "<categoria POI normalizada sem acentos, ou null>"
}}

Regras:
- Para "poi": "category" = tipo (ex: "hospital", "farmacia"); "area" = RA/bairro se mencionado, senão null.
- Para "geocode": "target" = endereço ou nome do lugar completo.
- Para "draw"/"remove": "target" = nome da RA (normalizado para a lista acima).
- Se ambíguo entre "poi" e "draw", prefira "poi".
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_text},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    try:
        return json.loads(raw)
    except Exception:
        return {"action": "none", "target": "", "area": None, "category": None}


def execute_command(parsed):
    action = parsed.get("action", "none")
    target = parsed.get("target", "") or ""
    area = parsed.get("area", None)
    category = parsed.get("category", None) or target

    if action == "clear":
        for store in ["drawn_layers", "poi_layers", "pin_layers"]:
            st.session_state[store] = {}
        return "🗺️ Todas as camadas foram removidas do mapa."

    if action == "remove":
        key_norm = _norm(target)
        removed  = []
        for store in ["drawn_layers", "poi_layers", "pin_layers"]:
            for k in list(st.session_state[store].keys()):
                if _norm(k) == key_norm or key_norm in _norm(k):
                    del st.session_state[store][k]
                    removed.append(k)
        if removed:
            return f"🗑️ Camada(s) removida(s): {', '.join(removed)}."
        return f"⚠️ Nenhuma camada chamada **{target}** encontrada no mapa."

    if action == "draw":
        label, features = find_sectors(target)
        if not features:
            return (
                f"⚠️ Não encontrei setores para **{target}**.\n"
                "Tente o nome de uma RA (ex: Ceilândia, Taguatinga) "
                "ou um código de setor com 15 dígitos."
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
                sum(lats) / len(lats), sum(lons) / len(lons)
            ]
        return f"✅ **{label}** desenhado no mapa com {len(features)} setor(es) censitário(s)."

    if action == "poi":
        with st.spinner(f"Consultando OpenStreetMap: **{category}**..."):
            pois, icon, color, err = search_poi(category, area)

        if err:
            return f"⚠️ Erro ao consultar a Overpass API: `{err}`"
        if not pois:
            area_msg = f" em **{area}**" if area else " no DF"
            return (
                f"🔍 Nenhum resultado encontrado para **{category}**{area_msg}.\n"
                "Tente outra categoria (ex: hospital, farmacia, clinica, ubs)."
            )

        layer_label = f"{icon} {category.title()}"
        if area:
            layer_label += f" – {area.title()}"
        st.session_state["poi_layers"][layer_label] = {
            "pois": pois, "icon": icon, "color": color
        }

        lats = [p["lat"] for p in pois]
        lons = [p["lon"] for p in pois]
        st.session_state["map_center"] = [
            sum(lats) / len(lats), sum(lons) / len(lons)
        ]

        area_msg   = f" em **{area.title()}**" if area else " no DF"
        lista_txt  = "\n".join(
            f"- **{p['name']}**"
            + (f" — {p['address']}" if p["address"] else "")
            + (f" ☎ {p['phone']}"   if p["phone"]   else "")
            for p in pois[:10]
        )
        sufixo = "\n\n*(exibindo os primeiros 10)*" if len(pois) > 10 else ""
        return (
            f"📍 **{len(pois)} resultado(s)** para **{category.title()}**{area_msg}:\n\n"
            + lista_txt + sufixo
        )

    if action == "geocode":
        with st.spinner(f"Localizando **{target}**..."):
            results, err = geocode(target)

        if err:
            return f"⚠️ Erro ao consultar o Nominatim: `{err}`"
        if not results:
            return (
                f"⚠️ Não foi possível localizar **{target}** no DF.\n"
                "Tente incluir o nome da RA ou um endereço mais completo."
            )

        best  = results[0]
        label = target.title()
        st.session_state["pin_layers"][label] = {
            "lat":          best["lat"],
            "lon":          best["lon"],
            "display_name": best["display_name"],
        }
        st.session_state["map_center"] = [best["lat"], best["lon"]]

        short_name = best["display_name"].split(",")[0]
        return (
            f"📌 **{short_name}** localizado e marcado no mapa.\n"
            f"Coordenadas: `{best['lat']:.5f}, {best['lon']:.5f}`"
        )

    return None

def get_color(value):
    if value > 100: return "#800026"
    elif value > 50: return "#BD0026"
    elif value > 20: return "#E31A1C"
    elif value > 10: return "#FC4E2A"
    elif value > 5: return "#FD8D3C"
    elif value > 0: return "#FEB24C"
    else: return "#FFEDA0"

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

retriever = setup_retriever()

if "drawn_layers" not in st.session_state:
    st.session_state["drawn_layers"] = {}
if "poi_layers" not in st.session_state:
    st.session_state["poi_layers"] = {}
if "pin_layers" not in st.session_state:
    st.session_state["pin_layers"] = {}
if "map_center" not in st.session_state:
    st.session_state["map_center"]   = [-15.793889, -47.882778]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if action == "dengue":

    gj = attach_dengue_to_geojson()
    st.session_state["dengue_layer"] = gj
    return "🦟 Mapa de casos de dengue carregado por setor censitário."

POLY_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

col_map, col_chat = st.columns([1, 1])

with col_map:
    st.subheader("Mapa do DF")

    all_labels = (
        list(st.session_state["drawn_layers"].keys())
        + list(st.session_state["poi_layers"].keys())
        + list(st.session_state["pin_layers"].keys())
    )
    if all_labels:
        st.caption("**Camadas ativas:** " + " · ".join(all_labels))
        if st.button("🗑️ Limpar todas as camadas"):
            for store in ["drawn_layers", "poi_layers", "pin_layers"]:
                st.session_state[store] = {}
            st.rerun()

    if "dengue_layer" in st.session_state:
        folium.Choropleth(
            geo_data=st.session_state["dengue_layer"],
            data=None,
            columns=None,
            key_on="feature.properties.CD_SETOR",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Casos de Dengue",
            highlight=True,
        ).add_to(m)

    center = st.session_state["map_center"]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    for label, layer in st.session_state["drawn_layers"].items():
        color   = layer["color"]
        payload = {"type": "FeatureCollection", "features": layer["features"]}
        folium.GeoJson(
            payload,
            name=label,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 1.2,
                "fillOpacity": 0.30,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["CD_SETOR", "NM_SUBDIST"],
                aliases=["Setor:", "RA:"],
            ),
        ).add_to(m)

        folium.GeoJson(
            st.session_state["dengue_layer"],
            style_function=lambda feature: {
                "fillColor": get_color(feature["properties"]["dengue_casos"]),
                "color": "black",
                "weight": 0.3,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NM_SUBDIST", "dengue_casos"],
                aliases=["Região:", "Casos:"]
            )
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

    folium.Marker(
        [-15.793889, -47.882778],
        popup="Fiocruz Brasília",
        tooltip="Fiocruz Brasília",
        icon=folium.Icon(color="red", icon="plus-sign"),
    ).add_to(m)

    if len(all_labels) > 1:
        folium.LayerControl().add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])


with col_chat:
    st.subheader("Assistente")

    for msg in st.session_state["chat_history"]:
        role_label = "🧑 Você" if msg["role"] == "user" else "🤖 Agente"
        with st.chat_message(msg["role"]):
            st.markdown(f"**{role_label}:** {msg['content']}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Digite um comando ou pergunta:",
            placeholder="Ex: Hospitais em Ceilândia | Onde fica a ESCS?",
        )
        submitted = st.form_submit_button("Enviar ↩")

    if submitted and user_input.strip():
        user_msg = user_input.strip()
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_msg}
        )

        with st.spinner("Processando..."):
            parsed   = parse_command(user_msg)
            response = execute_command(parsed)
            if response is None:
                response = answer_health_question(user_msg)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

    if st.session_state["chat_history"]:
        if st.button("🧹 Limpar conversa"):
            st.session_state["chat_history"] = []
            st.rerun()

st.markdown("---")
st.markdown("© 2026 : Saruê - Fiocruz Brasília\nGrupo de Inteligência Computacional na Saúde (GICS)")
