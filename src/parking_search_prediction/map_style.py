"""Fonds de carte communs aux cartes HTML (folium) : choix du fond et réglette d'opacité.

``add_basemaps(fmap)`` ajoute trois fonds au choix (boutons radio du sélecteur de
couches) — swisstopo gris, swisstopo couleur, OpenStreetMap, CARTO Voyager — et une réglette en
haut à gauche qui règle l'opacité du fond affiché (les couches de données ne
changent pas). Appeler ``folium.LayerControl`` après, comme d'habitude.
"""

from __future__ import annotations

SWISSTOPO_GREY = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"
SWISSTOPO_COLOUR = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg"
CARTO_VOYAGER = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"

BASEMAPS = {
    "swisstopo gris": (SWISSTOPO_GREY, "© swisstopo"),
    "swisstopo couleur": (SWISSTOPO_COLOUR, "© swisstopo"),
    "OpenStreetMap": ("OpenStreetMap", None),
    "CARTO Voyager": (CARTO_VOYAGER, "© OpenStreetMap contributors © CARTO"),
}


def add_basemaps(fmap, default: str = "swisstopo gris", opacity: float = 1.0):
    """Ajoute les fonds de carte (un seul visible à la fois) et la réglette d'opacité."""
    import folium
    from branca.element import MacroElement, Template

    order = [default, *[name for name in BASEMAPS if name != default]]
    for name in order:
        url, attribution = BASEMAPS[name]
        folium.TileLayer(url, attr=attribution, name=name, overlay=False, control=True,
                         show=name == default, opacity=opacity, max_zoom=19).add_to(fmap)

    macro = MacroElement()
    macro._template = Template(
        "{% macro script(this, kwargs) %}"
        "(function(){"
        "var map = {{ this._parent.get_name() }};"
        "var value = " + str(float(opacity)) + ";"
        "function apply(){ map.eachLayer(function(layer){ if (layer instanceof L.TileLayer) { layer.setOpacity(value); } }); }"
        "var control = L.control({position: 'topleft'});"
        "control.onAdd = function(){"
        "  var div = L.DomUtil.create('div', 'basemap-opacity');"
        "  div.style.cssText = 'background:#ffffffee;padding:6px 8px;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px #0003;';"
        "  div.innerHTML = 'Opacité du fond <input type=\"range\" min=\"0\" max=\"100\" value=\"' + Math.round(value*100) + '\" style=\"width:110px;vertical-align:middle\"> <span>' + Math.round(value*100) + ' %</span>';"
        "  L.DomEvent.disableClickPropagation(div); L.DomEvent.disableScrollPropagation(div);"
        "  var input = div.querySelector('input'), label = div.querySelector('span');"
        "  input.addEventListener('input', function(){ value = input.value / 100; label.textContent = input.value + ' %'; apply(); });"
        "  return div; };"
        "control.addTo(map);"
        "map.on('baselayerchange', apply);"
        "apply();"
        "})();"
        "{% endmacro %}"
    )
    fmap.add_child(macro)
    return fmap
