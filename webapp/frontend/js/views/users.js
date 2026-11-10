import { api, getMeta, h, fmt, withStates } from "../app.js";
import { bars } from "../charts.js";
import { baseMap, cssVar } from "./map.js";
import { pager } from "./restaurants.js";

const ACTIVITY = [["", "Any activity"], ["none", "No orders"], ["light", "1–3 orders"], ["regular", "4–15 orders"],
  ["heavy", "16+ orders"], ["traveller", "Ordered in the other city"]];

export async function render(root, { params, signal }) {
  if (params[0]) return detail(root, Number(params[0]), signal);
  const meta = await getMeta();
  const state = { city: "", activity: "", offset: 0 };
  const list = h("div", { class: "card" });
  const load = () => withStates(list, () => {
    const p = { limit: 50, offset: state.offset };
    if (state.city) p.city = state.city;
    if (state.activity) p.activity = state.activity;
    return api(`/api/users?${new URLSearchParams(p)}`, { signal });
  }, (d) => {
    list.append(h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["User", "Home", "Signed up", "Platform", "Age", "Orders", "In other city", "Last order"].map((x) => h("th", {}, x)))),
      h("tbody", {}, d.rows.map((u) => h("tr", {},
        h("td", {}, h("a", { href: `#/users/${u.id}` }, `#${u.id}`)), h("td", {}, `${u.district}, ${u.home_city}`),
        h("td", {}, fmt.date(u.signup_date)), h("td", {}, u.platform), h("td", {}, u.age_band),
        h("td", { class: "num" }, fmt.int(u.orders)), h("td", { class: "num" }, u.orders_away || ""), h("td", {}, fmt.dt(u.last_order))))))),
    pager(d.total, state, load));
  });
  const set = (k) => (e) => { state[k] = e.target.value; state.offset = 0; load(); };
  root.append(h("div", { class: "page-head" }, h("h1", {}, "Users"),
    h("p", { class: "lead" }, "6,000 users. Some order every week, some rarely, some never ordered (cold start), and about one in ten visits the other city.")),
  h("div", { class: "filters" },
    h("select", { onchange: set("city"), "aria-label": "City" }, h("option", { value: "" }, "Both cities"), meta.cities.map((c) => h("option", { value: c.code }, c.name))),
    h("select", { onchange: set("activity"), "aria-label": "Activity" }, ACTIVITY.map(([v, t]) => h("option", { value: v }, t)))), list);
  load();
}

function count(rows, key) {
  const m = new Map();
  for (const r of rows) m.set(r[key], (m.get(r[key]) || 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
}

async function detail(root, id, signal) {
  await withStates(root, () => Promise.all([api(`/api/users/${id}`, { signal }), getMeta()]), ([u, meta]) => {
    const label = Object.fromEntries(meta.cuisines.map((c) => [c.key, c.label]));
    const away = u.orders.filter((o) => o.city !== u.home_city).length;
    const mapEl = h("div", { class: "map small" }), cuisineEl = h("div"), dayEl = h("div");
    root.append(h("p", {}, h("a", { href: "#/users" }, "← Users")),
      h("div", { class: "page-head" }, h("h1", {}, `User #${u.id}`),
        h("p", { class: "lead" }, `${u.age_band}, ${u.platform}, lives in ${u.district} (${u.home_city}), signed up ${fmt.date(u.signup_date)} via ${u.channel.replace("_", " ")}.`)),
      h("div", { class: "tiles" },
        ...[[fmt.int(u.orders.length), "orders"], [fmt.int(u.sessions.sessions), "sessions"], [fmt.pct(u.sessions.conversion || 0, 0), "sessions that order"],
          [u.orders.length ? fmt.eur(u.orders.reduce((s, o) => s + Number(o.subtotal), 0) / u.orders.length) : "–", "average basket"],
          [fmt.int(away), "orders in the other city"]].map(([v, k]) => h("div", { class: "card tile" }, h("div", { class: "v" }, v), h("div", { class: "k" }, k)))),
      h("div", { class: "grid g2" }, h("div", { class: "card" }, h("h2", {}, "Where they order"), mapEl,
        h("p", { class: "muted small" }, "Pin = home, violet ring = work, dots = order locations (orange = in the other city).")),
      h("div", { class: "card" }, h("h2", {}, "Cuisines ordered"), cuisineEl, h("h2", { style: "margin-top:14px" }, "When"), dayEl)));

    const truth = u.truth;
    if (truth) {
      root.append(h("div", { class: "card truth", style: "margin-top:16px" },
        h("h2", {}, "Hidden ground truth (simulation only; models never see this)"),
        h("div", { class: "grid g3" },
          h("div", {}, h("h3", {}, "Favourite cuisines"), truth.top_cuisines.map((c) => h("div", {}, `${c.label} `, h("span", { class: "muted" }, fmt.pct(c.share, 0))))),
          h("dl", { class: "kv" },
            h("dt", {}, "Lifecycle"), h("dd", {}, truth.segment.replaceAll("_", " ")),
            h("dt", {}, "Vegetarian"), h("dd", {}, truth.vegetarian ? "yes" : "no"),
            h("dt", {}, "Avoids spicy"), h("dd", {}, truth.avoids_spicy ? "yes" : "no"),
            h("dt", {}, "Normal price"), h("dd", {}, fmt.eur(truth.reference_price_eur))),
          h("dl", { class: "kv" },
            h("dt", {}, "Price sensitivity"), h("dd", {}, `more than ${truth.price_sensitivity_pct}% of users`),
            h("dt", {}, "Reorder habit"), h("dd", {}, `stronger than ${truth.habit_pct}% of users`),
            h("dt", {}, "Sessions / week"), h("dd", {}, truth.sessions_per_week),
            h("dt", {}, "Trips"), h("dd", {}, truth.trips.length ? truth.trips.map((t) => `${t.start} (${t.days}d)`).join(", ") : "none")))));
    }

    const tl = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Order history (newest first)"));
    tl.append(u.orders.length ? h("div", { class: "timeline" }, u.orders.map((o) => h("div", { class: "order" },
      h("div", {}, h("b", {}, fmt.dt(o.ts)), " · ", h("a", { href: `#/restaurants/${o.restaurant_id}` }, o.restaurant), " ",
        h("span", { class: "badge" }, label[o.cuisine]), " ", o.city !== u.home_city ? h("span", { class: "badge" }, `in ${o.city}`) : null,
        " ", h("span", { class: "muted small" }, `${o.location_kind} · ${o.distance_km} km · ${fmt.eur(o.subtotal)}`)),
      h("div", { class: "muted small" }, o.items.map((i) => `${i.qty > 1 ? i.qty + "× " : ""}${i.name}`).join(", ")))))
      : h("p", { class: "muted" }, "No orders yet: a cold-start user."));
    root.append(tl);

    const cu = count(u.orders, "cuisine").slice(0, 8);
    if (cu.length) bars(cuisineEl, { horizontal: true, categories: cu.map(([k]) => label[k]), series: [{ name: "Orders", values: cu.map(([, v]) => v), color: "var(--s1)" }], format: fmt.int });
    else cuisineEl.append(h("p", { class: "muted" }, "No orders."));
    const hrs = [...Array(24).keys()];
    bars(dayEl, { categories: hrs.map((x) => `${x}h`), series: [{ name: "Orders", values: hrs.map((x) => u.orders.filter((o) => Number(o.ts.slice(11, 13)) === x).length), color: "var(--s1)" }], format: fmt.int, height: 150 });

    const map = baseMap(mapEl, [u.home_lat, u.home_lon], 12);
    const pts = [[u.home_lat, u.home_lon]];
    for (const o of u.orders) {
      L.circleMarker([o.lat, o.lon], { radius: 4, stroke: false, fillColor: cssVar(o.city !== u.home_city ? "--s2" : "--s1"), fillOpacity: 0.6 }).addTo(map);
      pts.push([o.lat, o.lon]);
    }
    L.marker([u.home_lat, u.home_lon], { title: "Home" }).addTo(map).bindPopup("Home");
    if (u.work_lat) { L.circleMarker([u.work_lat, u.work_lon], { radius: 8, color: cssVar("--s7"), weight: 3, fill: false }).addTo(map).bindPopup("Work"); pts.push([u.work_lat, u.work_lon]); }
    setTimeout(() => { map.invalidateSize(); if (pts.length > 1) map.fitBounds(pts, { padding: [20, 20], maxZoom: 14 }); }, 50);
  });
}
