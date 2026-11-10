import { api, getMeta, h, fmt, withStates, debounce, familyColor } from "../app.js";
import { bars } from "../charts.js";
import { baseMap, cssVar } from "./map.js";

const PAGE = 50;

export async function render(root, { params, signal }) {
  if (params[0]) return detail(root, Number(params[0]), signal);
  const meta = await getMeta();
  const state = { city: "", cuisine: "", q: "", offset: 0 };
  const label = Object.fromEntries(meta.cuisines.map((c) => [c.key, c.label]));
  const list = h("div", { class: "card" });
  const load = () => withStates(list, () => api(`/api/restaurants?${new URLSearchParams({
    ...(state.city && { city: state.city }), ...(state.cuisine && { cuisine: state.cuisine }),
    ...(state.q && { q: state.q }), limit: PAGE, offset: state.offset })}`, { signal }), (d) => table(d));
  const set = (k) => (e) => { state[k] = e.target.value; state.offset = 0; load(); };

  root.append(h("div", { class: "page-head" }, h("h1", {}, "Restaurants"),
    h("p", { class: "lead" }, "400 restaurants: independents plus outlets of seven fictional chains that exist in both cities.")),
  h("div", { class: "filters" },
    h("select", { onchange: set("city"), "aria-label": "City" }, h("option", { value: "" }, "Both cities"),
      meta.cities.map((c) => h("option", { value: c.code }, c.name))),
    h("select", { onchange: set("cuisine"), "aria-label": "Cuisine" }, h("option", { value: "" }, "All cuisines"),
      meta.cuisines.map((c) => h("option", { value: c.key }, c.label))),
    h("input", { type: "search", placeholder: "Search name", oninput: debounce(set("q")) })), list);

  function table(d) {
    list.append(h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Restaurant", "City", "District", "Cuisine", "Price", "Rating", "Dishes", "Orders"].map((x) => h("th", {}, x)))),
      h("tbody", {}, d.rows.map((r) => h("tr", {},
        h("td", {}, h("a", { href: `#/restaurants/${r.id}` }, r.name), r.chain ? [" ", h("span", { class: "badge" }, "chain")] : null),
        h("td", {}, r.city), h("td", {}, r.district),
        h("td", {}, h("span", { class: "dot", style: `background:${familyColor(meta.cuisines.find((c) => c.key === r.cuisine).family, meta.families)}` }), label[r.cuisine]),
        h("td", {}, "€".repeat(r.price_tier)), h("td", { class: "num" }, r.rating), h("td", { class: "num" }, r.items),
        h("td", { class: "num" }, fmt.int(r.orders))))))),
    pager(d.total, state, load));
  }
  load();
}

export function pager(total, state, load) {
  return h("div", { class: "pager" }, h("span", { class: "muted" }, `${fmt.int(state.offset + 1)}–${fmt.int(Math.min(state.offset + PAGE, total))} of ${fmt.int(total)}`),
    h("button", { disabled: state.offset === 0, onclick: () => { state.offset -= PAGE; load(); } }, "Previous"),
    h("button", { disabled: state.offset + PAGE >= total, onclick: () => { state.offset += PAGE; load(); } }, "Next"));
}

async function detail(root, id, signal) {
  await withStates(root, () => Promise.all([api(`/api/restaurants/${id}`, { signal }), getMeta()]), ([r, meta]) => {
    const label = Object.fromEntries(meta.cuisines.map((c) => [c.key, c.label]));
    const hours = r.hours.map(([o, c]) => `${o}:00–${c % 24}:00`).join(", ");
    const mapEl = h("div", { class: "map small" });
    const chart = h("div");
    root.append(h("p", {}, h("a", { href: "#/restaurants" }, "← Restaurants")),
      h("div", { class: "page-head" }, h("h1", {}, r.name), h("p", { class: "lead" }, r.tagline || "")),
      h("div", { class: "grid g3" },
        h("div", { class: "card" }, h("dl", { class: "kv" },
          h("dt", {}, "Cuisine"), h("dd", {}, label[r.cuisine]), h("dt", {}, "Where"), h("dd", {}, `${r.district}, ${r.city}`),
          h("dt", {}, "Price"), h("dd", {}, "€".repeat(r.price_tier)), h("dt", {}, "Rating"), h("dd", {}, `★ ${r.rating} (${fmt.int(r.n_ratings)})`),
          h("dt", {}, "Open"), h("dd", {}, hours), h("dt", {}, "Delivers"), h("dd", {}, `${r.delivery_radius_km} km`),
          h("dt", {}, "Prep"), h("dd", {}, `${r.prep_min} min`), r.chain ? [h("dt", {}, "Chain"), h("dd", {}, "yes, same menu in both cities")] : null)),
        h("div", { class: "card" }, h("h2", {}, "Orders by hour"), chart),
        h("div", { class: "card" }, mapEl)));
    const menu = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, `Menu (${r.menu.length})`));
    for (const cat of meta.categories) {
      const its = r.menu.filter((i) => i.category === cat);
      if (!its.length) continue;
      menu.append(h("div", { class: "menu-cat" }, h("h3", {}, `${cat[0].toUpperCase()}${cat.slice(1)}s`), its.map((i) => h("div", { class: "menu-item" },
        h("div", {}, h("div", {}, i.name, " ", i.vegetarian ? h("span", { class: "badge" }, i.vegan ? "vegan" : "veg") : null, " ",
          i.spicy ? h("span", { class: "badge" }, "spicy") : null), h("div", { class: "d" }, i.description)),
        h("div", { class: "num", style: "white-space:nowrap" }, fmt.eur(i.price), h("div", { class: "muted small" }, `${fmt.int(i.orders)} sold`))))));
    }
    root.append(menu);
    const hrs = [...Array(24).keys()];
    bars(chart, { categories: hrs.map((x) => `${x}h`), series: [{ name: "Orders", values: hrs.map((x) => r.orders_by_hour.find((o) => o.hour === x)?.orders || 0), color: "var(--s1)" }], format: fmt.int, height: 180 });
    const map = baseMap(mapEl, [r.lat, r.lon], 13);
    L.circle([r.lat, r.lon], { radius: r.delivery_radius_km * 1000, color: cssVar("--accent"), weight: 1, fillOpacity: 0.06 }).addTo(map);
    L.circleMarker([r.lat, r.lon], { radius: 7, color: cssVar("--surface"), weight: 2, fillColor: cssVar("--accent"), fillOpacity: 1 }).addTo(map);
    setTimeout(() => map.invalidateSize(), 50);
  });
}
