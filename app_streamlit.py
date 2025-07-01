import streamlit as st
from streamlit_folium import st_folium
import folium
import streamlit.components.v1 as components

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
        st.markdown("### Assistente Virtual")

        components.html("""
        <div id="webchat" style="width: 100%; height: 500px;"></div>
        <script src="https://cdn.botpress.cloud/webchat/v3.0/inject.js"></script>
        <script>
        window.botpress.on("webchat:ready", () => {
            window.botpress.open();
        });
        window.botpress.init({
            "botId": "8a3fd55b-b5f6-423c-a209-3613516526d2",
            "configuration": {},
            "clientId": "6c132f0e-1d4b-4d30-b068-213c294322cb",
            "selector": "#webchat"
        });
        </script>
        """, height=520)

st.markdown("---")
st.markdown("© 2025 - Saruê - Fiocruz Brasília  \nGrupo de Inteligência Computacional na Saúde (GICS)")
