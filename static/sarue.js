let map;
let ubsData = [];
let currentMarker = null;

function initMap() {
  map = L.map('map').setView([-15.793889, -47.882778], 11);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);
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

document.addEventListener('DOMContentLoaded', function () {
  initMap();
  carregarUBSeDropdown();

  /*d3.select('.bottom-right')
    .append('p')
    .text('Visualização interativa em breve...');*/
});
