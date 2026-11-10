import { api, getMeta, h, fmt, withStates, familyColor } from "../app.js";

const TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const ATTRIB = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function baseMap(el, center, zoom) {
  const map = L.map(el, { preferCanvas: true }).setView(center, zoom);
  L.tileLayer(TILE_URL, { maxZoom: 18, attribution: ATTRIB }).addTo(map);
  return map;
}

export async function render(root, { signal }) {
  root.append(h("div", { class: "page-head" }, h("h1", {}, "Map"),
    h("p", { class: "lead" }, "Restaurants coloured by cuisine family and sized by orders; user homes as small dots. " +
      "Click anywhere to see which restaurants deliver to that point (PostGIS distance against each restaurant's delivery radius).")));
  const body = h("div");
  root.append(body);
  await withStates(body, () => Promise.all([getMeta(), api("/api/map/restaurants", { signal }), api("/api/map/users", { signal })]),
    ([meta, rests, users]) => draw(body, meta, rests, users, signal));
}

function draw(root, meta, rests, users, signal) {
  const family = Object.fromEntries(meta.cuisines.map((c) => [c.key, c.family]));
  const label = Object.fromEntries(meta.cuisines.map((c) => [c.key, c.label]));
  const shown = new Set(meta.families);
  let city = "BCN", showUsers = true;

  const citySeg = h("div", { class: "seg" });
  const mapEl = h("div", { class: "map" });
  const side = h("div", { class: "card" }, h("h2", {}, "Delivers here"), h("p", { class: "muted" }, "Click the map."));
  const legend = h("div", { class: "legend" });
  root.append(h("div", { class: "filters" }, citySeg,
    h("label", {}, h("input", { type: "checkbox", checked: true, onchange: (e) => { showUsers = e.target.checked; redraw(); } }), " users")),
    legend, h("div", { class: "grid", style: "grid-template-columns: minmax(0,3fr) minmax(0,1fr)" }, mapEl, side));

  const c0 = meta.cities[0];
  const map = baseMap(mapEl, [c0.lat, c0.lon], c0.zoom);
  const rLayer = L.layerGroup().addTo(map), uLayer = L.layerGroup().addTo(map);
  let pick = null;
  const maxOrders = Math.max(...rests.map((r) => r.orders), 1);

  for (const c of meta.cities) {
    citySeg.append(h("button", { type: "button", class: c.code === city ? "on" : "", onclick: (e) => {
      city = c.code; citySeg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === e.target));
      map.setView([c.lat, c.lon], c.zoom);
    } }, c.name));
  }
  for (const f of meta.families) {
    legend.append(h("label", {}, h("input", { type: "checkbox", checked: true, onchange: (e) => {
      e.target.checked ? shown.add(f) : shown.delete(f); redraw();
    } }), " ", h("span", { class: "dot", style: `background:${familyColor(f, meta.families)}` }), f));
  }

  function redraw() {
    rLayer.clearLayers(); uLayer.clearLayers();
    const surface = cssVar("--surface"), userColor = cssVar("--muted");
    if (showUsers) {
      for (const u of users) L.circleMarker([u.lat, u.lon], { radius: 2, stroke: false, fillColor: userColor, fillOpacity: 0.35 }).addTo(uLayer);
    }
    for (const r of rests) {
      const f = family[r.cuisine];
      if (!shown.has(f)) continue;
      const color = cssVar(`--s${meta.families.indexOf(f) + 1}`);
      const m = L.circleMarker([r.lat, r.lon], { radius: 4 + 8 * Math.sqrt(r.orders / maxOrders), color: surface, weight: 2,
        fillColor: color, fillOpacity: 0.9 });
      m.bindPopup(() => h("div", {}, h("b", {}, r.name), h("br"), `${label[r.cuisine]} · ${"€".repeat(r.price_tier)} · ★ ${r.rating}`,
        h("br"), `${fmt.int(r.orders)} orders · delivers ${r.delivery_radius_km} km`, r.chain ? h("div", {}, h("span", { class: "badge" }, "chain")) : null,
        h("div", {}, h("a", { href: `#/restaurants/${r.id}` }, "Open menu →"))));
      m.addTo(rLayer);
    }
  }
  redraw();

  map.on("click", async (e) => {
    const { lat, lng } = e.latlng;
    pick?.remove();
    pick = L.marker([lat, lng]).addTo(map);
    side.replaceChildren(h("h2", {}, "Delivers here"), h("p", { class: "muted" }, "Loading…"));
    try {
      const rows = await api(`/api/map/near?lat=${lat.toFixed(5)}&lon=${lng.toFixed(5)}`, { signal });
      side.replaceChildren(h("h2", {}, "Delivers here"),
        h("p", { class: "muted small" }, `${rows.length} restaurants deliver to ${lat.toFixed(4)}, ${lng.toFixed(4)}.`),
        h("div", { class: "timeline" }, rows.length ? rows.map((r) => h("div", { class: "order" },
          h("a", { href: `#/restaurants/${r.id}` }, r.name), h("div", { class: "muted small" }, `${label[r.cuisine]} · ${r.km} km`)))
          : h("p", {}, "Nothing delivers here.")));
    } catch {
      side.replaceChildren(h("h2", {}, "Delivers here"), h("p", { class: "state error" }, "Could not load."));
    }
  });
  window.addEventListener("themechange", redraw, { signal });
  setTimeout(() => map.invalidateSize(), 50);
}
