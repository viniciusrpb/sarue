let map;
let ubsData = [];
let currentMarker = null;

/*https://jonnymccullagh.github.io/leaflet-examples/map-satellite.html*/
function initMap() {
  map = L.map('map', {
    center: [-15.793889, -47.882778],
    zoom: 11,
    layers: []
  });

  const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  });

  const satelliteLayer = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles © Esri'
  });

  osmLayer.addTo(map);

  const baseMaps = {
    "Mapa padrão": osmLayer,
    "Satélite": satelliteLayer
  };

  L.control.layers(baseMaps).addTo(map);
}

function carregarUBSeDropdown() {
  fetch('/api/ubs')
    .then(response => response.json())
    .then(data => {
      ubsData = data;
      const select = document.getElementById('ubs-select');

      data.forEach(ubs => {
        const option = document.createElement('option');
        option.value = ubs.nome;
        option.textContent = ubs.nome;
        select.appendChild(option);
      });

      select.addEventListener('change', function () {
        const nomeSelecionado = this.value;
        const ubsSelecionada = ubsData.find(u => u.nome === nomeSelecionado);

        if (ubsSelecionada) {
          const { latitude, longitude, nome } = ubsSelecionada;

          if (currentMarker) {
            map.removeLayer(currentMarker);
          }

          currentMarker = L.marker([latitude, longitude])
            .addTo(map)
            .bindPopup(`<strong>${nome}</strong><br>${ubsSelecionada.endereco}`)
            .openPopup();

          map.setView([latitude, longitude], 15);
        }
      });
    })
    .catch(error => console.error('Erro ao carregar UBS:', error));
}

document.getElementById('load-info-btn').addEventListener('click', async function () {
  const selectedUBS = document.getElementById('ubs-select').value;
  const tipoSelecionado = document.querySelector('input[name="info-type"]:checked');

  if (!selectedUBS || !tipoSelecionado) {
    alert("Selecione uma UBS e um tipo de informação.");
    return;
  }

  const tipo = tipoSelecionado.value;

  const ubsSelecionada = ubsData.find(u => u.nome === selectedUBS);
  if (!ubsSelecionada) {
    alert("Coordenadas da UBS não encontradas.");
    return;
  }

  const { latitude, longitude } = ubsSelecionada;

  if (currentMarker) map.removeLayer(currentMarker);

  let tooltipText = `<strong>${selectedUBS}</strong><br>`;

  if (tipo === "oficiais") {
    const subtipos = Array.from(document.querySelectorAll('input[name="subtipo"]:checked'))
                          .map(cb => cb.value);

    if (subtipos.length === 0) {
      alert("Selecione pelo menos uma categoria de publicação oficial.");
      return;
    }

    const response = await fetch('/api/ubs-info');
    const data = await response.json();
    const ubsInfo = data[selectedUBS];

    if (!ubsInfo) {
      tooltipText += "Nenhuma publicação oficial encontrada.";
    } else {
      subtipos.forEach(type => {
        if (ubsInfo[type]) {
          tooltipText += `<em>${type.charAt(0).toUpperCase() + type.slice(1)}:</em> ${ubsInfo[type]}<br>`;
        }
      });
    }

  } else if (tipo === "noticias") {
    const response = await fetch('database/noticias.json');
    const noticias = await response.json();
    const lista = noticias[selectedUBS] || [];

    if (lista.length > 0) {
      tooltipText += `<em>Notícias:</em><br>`;
      lista.forEach(n => {
        tooltipText += `• <a href="${n.link}" target="_blank">${n.titulo}</a><br>`;
      });
    } else {
      tooltipText += "Nenhuma notícia encontrada para esta UBS.";
    }
  }

  currentMarker = L.marker([latitude, longitude])
    .addTo(map)
    .bindPopup(tooltipText)
    .openPopup();

  map.setView([latitude, longitude], 15);
});

function initBotpressWebChat() {
  const script = document.createElement('script');
  script.src = "https://cdn.botpress.cloud/webchat/v3.0/inject.js";
  script.onload = () => {
    window.botpress.on("webchat:ready", () => {
      window.botpress.open();
    });

    window.botpress.init({
      "botId": "8a3fd55b-b5f6-423c-a209-3613516526d2",
      "configuration": {
        "website": {},
        "email": {},
        "phone": {},
        "termsOfService": {},
        "privacyPolicy": {}
      },
      "clientId": "6c132f0e-1d4b-4d30-b068-213c294322cb",
      "selector": "#webchat"
    });
  };

  document.head.appendChild(script);

  const style = document.createElement('style');
  style.textContent = `
    #webchat .bpWebchat {
      position: unset;
      width: 100%;
      height: 100%;
      max-height: 100%;
      max-width: 100%;
    }
    #webchat .bpFab {
      display: none;
    }
  `;
  document.head.appendChild(style);
}


document.addEventListener('DOMContentLoaded', function () {
  initMap();
  carregarUBSeDropdown();
  initBotpressWebChat();
  /*d3.select('.bottom-right')
    .append('p')
    .text('Visualização interativa em breve...');*/
});
