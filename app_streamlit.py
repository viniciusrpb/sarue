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
    "clinic": ("amenity", "clinic", "🏥", "blue"),
    "farmacia": ("amenity", "pharmacy", "💊", "green"),
    "farmácia": ("amenity", "pharmacy", "💊", "green"),
    "pharmacy": ("amenity", "pharmacy", "💊", "green"),
    "medico": ("amenity", "doctors", "👨‍⚕️", "cadetblue"),
    "médico": ("amenity", "doctors", "👨‍⚕️", "cadetblue"),
    "doctor": ("amenity", "doctors", "👨‍⚕️", "cadetblue"),
    "dentista": ("amenity", "dentist", "🦷", "purple"),
    "dentist": ("amenity", "dentist", "🦷", "purple"),
    "social": ("amenity", "social_facility", "🤝", "orange"),
    "cras": ("amenity", "social_facility", "🤝", "orange"),
    "creas": ("amenity", "social_facility", "🤝", "orange"),
    "social facility": ("amenity", "social_facility", "🤝", "orange"),
    "escola": ("amenity", "school", "🏫", "darkblue"),
    "school": ("amenity", "school", "🏫", "darkblue"),
    "creche": ("amenity", "kindergarten", "👶", "pink"),
    "kindergarten": ("amenity", "kindergarten", "👶", "pink"),
    "saude": ("healthcare", "*", "➕", "red"),
    "saúde": ("healthcare", "*", "➕", "red"),
    "health": ("healthcare", "*", "➕", "red"),
    "healthcare": ("healthcare", "*", "➕", "red"),
    "academia": ("leisure", "fitness_centre", "🏋️", "darkgreen"),
    "gym": ("leisure", "fitness_centre", "🏋️", "darkgreen"),
    "fitness": ("leisure", "fitness_centre", "🏋️", "darkgreen"),
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
            text    = f"{titulo}\n\n{noticia}".strip()
            if len(text) > 50:
                documents.append(Document(page_content=text, metadata={"source": path}))
    return documents


@st.cache_data
def load_dengue_data():
    path = os.path.join(os.path.dirname(__file__), "database", "dados_dengue-16042026-ano_2026.csv")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def aggregate_dengue_by_sector(df):
    grouped = df.groupby("cd_setor")["casos"].sum().reset_index()
    return grouped


def attach_dengue_to_geojson():
    gj, by_code, _ = load_geojson()
    df  = load_dengue_data()
    agg = aggregate_dengue_by_sector(df)

    dengue_map = dict(zip(agg["cd_setor"], agg["casos"]))

    for feat in gj["features"]:
        code = feat["properties"].get("CD_SETOR")
        feat["properties"]["dengue_casos"] = dengue_map.get(code, 0)

    return gj


@st.cache_resource(show_spinner=False)
def setup_retriever():
    docs     = load_documents()
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks   = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="juridics/bertimbau-base-portuguese-sts-scale")
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
        return [], "Connection timeout. Please try again."
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
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers=HEADERS_OSM,
            timeout=30,
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except requests.exceptions.Timeout:
        return [], icon, color, "Connection timeout while querying Overpass API."
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
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Detect the language of the user message. "
                    "Reply with only 'pt' if it is Portuguese, or 'en' if it is English or any other language."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    lang = resp.choices[0].message.content.strip().lower()
    return "pt" if "pt" in lang else "en"


def parse_command(user_text):
    subdist_str  = ", ".join(get_subdist_list())
    poi_cats_str = ", ".join(sorted({_norm(k) for k in OSM_CATEGORIES}))

    system = f"""You are the map control agent for the Saruê app (Fiocruz Brasília, DF, Brazil).
Classify the user message and respond ONLY with pure JSON (no markdown, no explanation).

Actions:
- "draw": draw census sector polygons for an Administrative Region (RA) on the map
- "remove": remove a specific layer from the map
- "clear": remove ALL layers from the map
- "poi": search for points of interest on OpenStreetMap (hospitals, pharmacies, clinics, etc.)
- "geocode": locate and pin a specific address or place on the map
- "dengue": display a dengue case choropleth map by census sector
- "none": public health question or topic unrelated to map control

Available RAs (for draw/remove, Portuguese names): {subdist_str}
Available POI categories: {poi_cats_str}

Response format:
{{
  "action":   "draw"|"remove"|"clear"|"poi"|"geocode"|"dengue"|"none",
  "target":   "<RA name, sector code, full address or category>",
  "area":     "<RA/neighbourhood name for POI search, or null>",
  "category": "<normalised POI category without accents, or null>"
}}

Rules:
- For "poi": "category" = type (e.g. "hospital", "farmacia", "pharmacy"); "area" = RA/neighbourhood if mentioned, else null.
- For "geocode": "target" = full address or place name.
- For "draw"/"remove": "target" = RA name (normalised to the list above).
- If ambiguous between "poi" and "draw", prefer "poi".
- The user may write in Portuguese or English; handle both.
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


def execute_command(parsed, lang="en"):
    action   = parsed.get("action", "none")
    target   = parsed.get("target", "") or ""
    area     = parsed.get("area", None)
    category = parsed.get("category", None) or target

    def msg(en, pt):
        return pt if lang == "pt" else en

    if action == "clear":
        for store in ["drawn_layers", "poi_layers", "pin_layers"]:
            st.session_state[store] = {}
        return msg(
            "🗺️ All layers have been removed from the map.",
            "🗺️ Todas as camadas foram removidas do mapa.",
        )

    if action == "remove":
        key_norm = _norm(target)
        removed  = []
        for store in ["drawn_layers", "poi_layers", "pin_layers"]:
            for k in list(st.session_state[store].keys()):
                if _norm(k) == key_norm or key_norm in _norm(k):
                    del st.session_state[store][k]
                    removed.append(k)
        if removed:
            return msg(
                f"🗑️ Layer(s) removed: {', '.join(removed)}.",
                f"🗑️ Camada(s) removida(s): {', '.join(removed)}.",
            )
        return msg(
            f"⚠️ No layer named **{target}** found on the map.",
            f"⚠️ Nenhuma camada chamada **{target}** encontrada no mapa.",
        )

    if action == "draw":
        label, features = find_sectors(target)
        if not features:
            return msg(
                (
                    f"⚠️ No sectors found for **{target}**.\n"
                    "Try the name of an Administrative Region (e.g. Ceilândia, Taguatinga) "
                    "or a 15-digit sector code."
                ),
                (
                    f"⚠️ Não encontrei setores para **{target}**.\n"
                    "Tente o nome de uma RA (ex: Ceilândia, Taguatinga) "
                    "ou um código de setor com 15 dígitos."
                ),
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
        return msg(
            f"✅ **{label}** drawn on the map with {len(features)} census sector(s).",
            f"✅ **{label}** desenhado no mapa com {len(features)} setor(es) censitário(s).",
        )

    if action == "poi":
        spinner_msg = (
            f"Querying OpenStreetMap: **{category}**..."
            if lang == "en"
            else f"Consultando OpenStreetMap: **{category}**..."
        )
        with st.spinner(spinner_msg):
            pois, icon, color, err = search_poi(category, area)

        if err:
            return msg(
                f"⚠️ Error querying the Overpass API: `{err}`",
                f"⚠️ Erro ao consultar a Overpass API: `{err}`",
            )
        if not pois:
            area_msg_en = f" in **{area}**" if area else " in the Federal District"
            area_msg_pt = f" em **{area}**" if area else " no DF"
            return msg(
                (
                    f"🔍 No results found for **{category}**{area_msg_en}.\n"
                    "Try another category (e.g. hospital, pharmacy, clinic, ubs)."
                ),
                (
                    f"🔍 Nenhum resultado encontrado para **{category}**{area_msg_pt}.\n"
                    "Tente outra categoria (ex: hospital, farmacia, clinica, ubs)."
                ),
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
            (
                f"📍 **{len(pois)} result(s)** for **{category.title()}**{area_msg_en}:\n\n"
                + lista_txt + suffix_en
            ),
            (
                f"📍 **{len(pois)} resultado(s)** para **{category.title()}**{area_msg_pt}:\n\n"
                + lista_txt + suffix_pt
            ),
        )

    if action == "dengue":
        gj = attach_dengue_to_geojson()
        st.session_state["dengue_layer"] = gj
        return msg(
            "🦟 Dengue case map loaded by census sector.",
            "🦟 Mapa de casos de dengue carregado por setor censitário.",
        )

    if action == "geocode":
        spinner_msg = (
            f"Locating **{target}**..."
            if lang == "en"
            else f"Localizando **{target}**..."
        )
        with st.spinner(spinner_msg):
            results, err = geocode(target)

        if err:
            return msg(
                f"⚠️ Error querying Nominatim: `{err}`",
                f"⚠️ Erro ao consultar o Nominatim: `{err}`",
            )
        if not results:
            return msg(
                (
                    f"⚠️ Could not locate **{target}** in the Federal District.\n"
                    "Try including the Administrative Region name or a more complete address."
                ),
                (
                    f"⚠️ Não foi possível localizar **{target}** no DF.\n"
                    "Tente incluir o nome da RA ou um endereço mais completo."
                ),
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
        return msg(
            (
                f"📌 **{short_name}** located and pinned on the map.\n"
                f"Coordinates: `{best['lat']:.5f}, {best['lon']:.5f}`"
            ),
            (
                f"📌 **{short_name}** localizado e marcado no mapa.\n"
                f"Coordenadas: `{best['lat']:.5f}, {best['lon']:.5f}`"
            ),
        )

    return None


def get_color(value):
    if value > 100: return "#800026"
    elif value > 50: return "#BD0026"
    elif value > 20: return "#E31A1C"
    elif value > 10: return "#FC4E2A"
    elif value > 5:  return "#FD8D3C"
    elif value > 0:  return "#FEB24C"
    else:            return "#FFEDA0"


def answer_health_question(pergunta, lang="en"):
    documentos = retriever.invoke(pergunta)
    MAX_CHARS  = 6000
    contexto   = ""
    for doc in documentos:
        if len(contexto) + len(doc.page_content) > MAX_CHARS:
            break
        contexto += doc.page_content + "\n\n---\n\n"

    lang_instruction = (
        "Answer in English, even though the context is in Portuguese. "
        "Translate any relevant information from the context as needed."
        if lang == "en"
        else "Responda em português."
    )

    prompt = f"""You are an assistant specialised in Brazilian public health.
Answer exclusively based on the information provided in the context below.
If the answer is not clearly present, say you could not find sufficient evidence.
{lang_instruction}

Context:
{contexto}

Question:
{pergunta}

Answer:"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical, objective public-health assistant. "
                    "Always reply in the same language as the user's question."
                ),
            },
            {"role": "user", "content": prompt},
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
    st.session_state["map_center"] = [-15.793889, -47.882778]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

POLY_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


col_map, col_chat = st.columns([1, 1])

with col_map:
    st.subheader("Map – Federal District (DF)")

    all_labels = (
        list(st.session_state["drawn_layers"].keys())
        + list(st.session_state["poi_layers"].keys())
        + list(st.session_state["pin_layers"].keys())
    )
    if all_labels:
        st.caption("**Active layers:** " + " · ".join(all_labels))
        if st.button("🗑️ Clear all layers"):
            for store in ["drawn_layers", "poi_layers", "pin_layers"]:
                st.session_state[store] = {}
            st.rerun()

    center = st.session_state["map_center"]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    if "dengue_layer" in st.session_state:
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
                aliases=["Region:", "Cases:"],
            ),
        ).add_to(m)

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
    st.subheader("Assistant")

    for msg_item in st.session_state["chat_history"]:
        role_label = "🧑 You" if msg_item["role"] == "user" else "🤖 Agent"
        with st.chat_message(msg_item["role"]):
            st.markdown(f"**{role_label}:** {msg_item['content']}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type a command or question:",
            placeholder="e.g. Hospitals in Ceilândia | Where is ESCS? | Dengue map",
        )
        submitted = st.form_submit_button("Send ↩")

    if submitted and user_input.strip():
        user_msg = user_input.strip()
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_msg}
        )

        with st.spinner("Processing..."):
            lang     = detect_language(user_msg)
            parsed   = parse_command(user_msg)
            response = execute_command(parsed, lang=lang)
            if response is None:
                response = answer_health_question(user_msg, lang=lang)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

    if st.session_state["chat_history"]:
        if st.button("🧹 Clear conversation"):
            st.session_state["chat_history"] = []
            st.rerun()

st.markdown("---")
st.markdown("© 2026 · Saruê – Fiocruz Brasília · Grupo de Inteligência Computacional na Saúde (GICS)")
