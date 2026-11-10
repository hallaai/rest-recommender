import { api, getMeta, h, fmt, withStates, debounce } from "../app.js";
import { baseMap, cssVar } from "./map.js";
import { algoName, ALGORITHMS } from "../algorithms.js";
import { showTip, hideTip } from "../charts.js";

const DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const ALGO_ORDER = ["pipeline", "buy_again", "als", "ease", "item_item", "geo_time_pop", "popularity", "random"];

export async function render(root, { signal }) {
  const meta = await getMeta();
  const ctx = { user: 3638, lat: null, lon: null, hour: 21, dow: 4, rain: false, hot: false, mode: "user" };
  root.append(h("div", { class: "page-head" }, h("h1", {}, "Recommend"),
    h("p", { class: "lead" }, "Live top-5 from every algorithm for one user, side by side. Pick a user (or click the map " +
      "for a brand-new user with no history), set the time and weather, and compare. “RE” marks a dish the user ordered before.")));

  const warm = h("div", { class: "note", style: "margin-bottom:12px" },
    "First run fits the models (ALS takes up to ~2 min). This happens once per server start.");
  root.append(warm);

  const controls = h("div", { class: "card", style: "margin-bottom:16px" });
  const results = h("div");
  root.append(controls, results);
  buildControls(controls, ctx, meta, () => run());

  let current = null;
  async function run() {
    current?.abort();
    current = new AbortController();
    const sig = current.signal;
    const url = ctx.mode === "user"
      ? `/api/recommend?user=${ctx.user}&hour=${ctx.hour}&dow=${ctx.dow}&rain=${ctx.rain}&hot=${ctx.hot}`
      : `/api/recommend/at?lat=${ctx.lat.toFixed(5)}&lon=${ctx.lon.toFixed(5)}&hour=${ctx.hour}&dow=${ctx.dow}&rain=${ctx.rain}&hot=${ctx.hot}`;
    await withStates(results, () => pollReady(sig).then(() => api(url, { signal: sig })),
      (d) => { warm.hidden = true; draw(results, d); });
  }
  run();
}

async function pollReady(signal) {
  for (let i = 0; i < 40; i++) {
    const s = await api("/api/recommend/status", { signal });
    if (s.status === "ready") return;
    if (s.status === "error") throw new Error("Model fitting failed on the server.");
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error("Models are taking too long to warm up.");
}

function buildControls(root, ctx, meta, run) {
  const userRow = h("div", { class: "filters" },
    h("span", { class: "muted" }, "User"),
    h("input", { type: "number", min: 0, value: ctx.user, style: "width:100px",
      oninput: debounce((e) => { ctx.user = Number(e.target.value); ctx.mode = "user"; run(); }) }),
    h("span", { class: "muted small" }, "or click the map below for a new user"));
  const timeRow = h("div", { class: "filters" },
    h("span", { class: "muted" }, "Time"),
    h("select", { onchange: (e) => { ctx.hour = Number(e.target.value); run(); } },
      [...Array(24).keys()].map((hh) => h("option", { value: hh, selected: hh === ctx.hour }, `${hh}:00`))),
    h("select", { onchange: (e) => { ctx.dow = Number(e.target.value); run(); } },
      DOW.map((d, i) => h("option", { value: i, selected: i === ctx.dow }, d))),
    h("label", {}, h("input", { type: "checkbox", onchange: (e) => { ctx.rain = e.target.checked; run(); } }), " rain"),
    h("label", {}, h("input", { type: "checkbox", onchange: (e) => { ctx.hot = e.target.checked; run(); } }), " hot (>30°)"));
  const mapEl = h("div", { class: "map small", style: "margin-top:10px" });
  root.append(userRow, timeRow, mapEl);
  const c0 = meta.cities[0];
  const map = baseMap(mapEl, [c0.lat, c0.lon], 12);
  let marker = null;
  map.on("click", (e) => {
    ctx.mode = "at"; ctx.lat = e.latlng.lat; ctx.lon = e.latlng.lng;
    marker?.remove(); marker = L.marker([ctx.lat, ctx.lon]).addTo(map);
    run();
  });
  setTimeout(() => map.invalidateSize(), 60);
}

function draw(root, d) {
  const c = d.context;
  root.append(h("div", { class: "note", style: "margin-bottom:12px" },
    `${c.user_known ? "User with history" : "New user (no history)"} · ${c.city} · ${c.daypart} ` +
    `· ${c.rain ? "rain" : "dry"}${c.hot ? ", hot" : ""} · ${fmt.int(c.n_candidates)} dishes available here now.`));
  if (!c.user_known) {
    root.append(h("p", { class: "muted small", style: "margin-bottom:12px" },
      "A new user has no order history, so buy-again, ALS, EASE and item-to-item have nothing to work from — only the " +
      "popularity sources produce useful picks. This is the cold-start case."));
  }
  const byName = Object.fromEntries(d.results.map((r) => [r.algorithm, r]));
  const hasOracle = c.has_oracle;

  // Colour a dish the SAME across every algorithm, so you can see which methods agree on it.
  // Only dishes that appear in more than one algorithm's list get a colour; unique dishes stay
  // neutral, so the colours mark exactly the overlap.
  const colorOf = buildColorMap(d.results);

  // The fused pipeline is THE recommendation — a full-width hero at the top.
  const fused = byName.pipeline;
  if (fused) {
    root.append(heroCard(fused, hasOracle, colorOf));
  } else if (c.pipeline_status && c.pipeline_status !== "ready") {
    const msg = c.pipeline_status === "error"
      ? "The fused pipeline could not be fitted on the server (see logs)."
      : "The fused pipeline (production path) is still warming up — it trains a ranker on startup. " +
        "Reload in ~30s to see the final recommendation here.";
    root.append(h("div", { class: "card", style: "margin-bottom:16px" },
      h("h2", { style: "margin:0 0 6px" }, "The recommendation (fused pipeline)"),
      h("p", { class: "muted small", style: "margin:0" }, msg)));
  }

  // Every individual method below — "how each method would have chosen on its own".
  root.append(h("h2", { style: "margin:20px 0 4px" }, "How each method would choose on its own"),
    h("p", { class: "muted small", style: "margin:0 0 12px" },
      "The fused pipeline above combines all of these. Each card shows the same five-slot screen that method would " +
      "produce alone, with its buy-probability — so you can see what the fusion is built from. " +
      "A dish that more than one method picks is given the same colour everywhere (with a coloured bar), so you can " +
      "spot where the algorithms agree; a dish only one method picks stays neutral." +
      (hasOracle ? "" : " (Live buy-probability needs the simulation's truth files, which are not loaded here.)")));
  const grid = h("div", { class: "grid", style: "grid-template-columns: repeat(auto-fill, minmax(320px, 1fr))" });
  for (const algo of ALGO_ORDER) {
    if (algo === "pipeline") continue;               // shown as the hero above
    const r = byName[algo];
    if (!r) continue;
    grid.append(algoCard(algo, r, hasOracle, colorOf));
  }
  root.append(grid);
}

// A stable colour for each dish that more than one algorithm recommends. Unique dishes -> null.
function buildColorMap(results) {
  const count = new Map();
  for (const r of results) for (const it of r.items || []) count.set(it.item_id, (count.get(it.item_id) || 0) + 1);
  // Distinct, theme-safe hues; assigned in order of first appearance for stability within a render.
  const palette = ["#e07a5f", "#3d9970", "#6a8caf", "#c9a227", "#9b5de5", "#2a9d8f",
                   "#d1495b", "#457b9d", "#b5651d", "#7678ed", "#587d71", "#bc4749"];
  const shared = [];
  for (const r of results) for (const it of r.items || []) {
    if (count.get(it.item_id) >= 2 && !shared.includes(it.item_id)) shared.push(it.item_id);
  }
  const map = new Map();
  shared.forEach((id, i) => map.set(id, palette[i % palette.length]));
  return (id) => map.get(id) || null;
}

// Two demand numbers for a result: the live buy-probability of the 5 shown (oracle, demo-only)
// and the method's average demand@5 over the October test set (from the offline eval).
function demandBadges(r, hasOracle) {
  const badges = [];
  if (hasOracle && r.live_demand != null) {
    const b = h("span", { class: "badge", style: "cursor:help;background:var(--s1);color:#fff;border-color:var(--s1)" },
      `buy-prob ${(r.live_demand * 100).toFixed(0)}%`);
    const lines = ["Live buy-probability (this screen)",
      "Chance this user orders one of the 5 shown,",
      "from the simulation's hidden truth. Demo-only:",
      "production can't see these probabilities."];
    b.addEventListener("mousemove", (e) => showTip(e, lines));
    b.addEventListener("mouseleave", hideTip);
    badges.push(b);
  }
  if (r.leaderboard_demand != null) {
    const b = h("span", { class: "badge", style: "cursor:help" }, `avg ${r.leaderboard_demand.toFixed(3)}`);
    const lines = ["Average Demand@5 (October test set)",
      "This method's measured quality over all test",
      "requests — a global number, not this request.",
      "The headline metric on the Compare page."];
    b.addEventListener("mousemove", (e) => showTip(e, lines));
    b.addEventListener("mouseleave", hideTip);
    badges.push(b);
  }
  return badges;
}

function dishList(items, colorOf) {
  const list = h("ol", { style: "margin:0;padding-left:20px" });
  for (const it of items) {
    const color = colorOf ? colorOf(it.item_id) : null;
    // A shared dish gets a coloured dot and a left rule in its colour, so the same product reads
    // the same across every algorithm.
    const dot = color ? h("span", { style:
      `display:inline-block;width:9px;height:9px;border-radius:50%;background:${color};margin-right:6px;vertical-align:middle` }) : null;
    const li = h("li", { style: "margin-bottom:8px;cursor:help" +
      (color ? `;border-left:3px solid ${color};padding-left:8px;margin-left:-8px;list-style:none` : "") },
      h("div", {}, dot, h("b", {}, it.name), it.is_reorder ? [" ", h("span", { class: "badge" }, "RE")] : null,
        " ", h("span", { class: "muted small" }, "ⓘ")),
      h("div", { class: "muted small" },
        h("a", { href: `#/restaurants/${it.restaurant_id}` }, it.restaurant), ` · ${fmt.eur(it.price)} · ${it.distance_km} km`),
      h("div", { class: "small", style: "color:var(--s1)" }, it.reason));
    if (it.details) {
      const lines = [it.name, ...Object.entries(it.details).map(([k, v]) => ["", k, String(v)])];
      li.addEventListener("mousemove", (e) => showTip(e, lines));
      li.addEventListener("mouseleave", hideTip);
    }
    list.append(li);
  }
  return list;
}

function heroCard(r, hasOracle, colorOf) {
  const info = ALGORITHMS.pipeline || {};
  const card = h("div", { class: "card", style: "border:2px solid var(--s1);margin-bottom:8px" });
  card.append(h("div", { style: "display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap" },
    h("h2", { style: "margin:0" }, "The recommendation ",
      h("span", { class: "badge", style: "background:var(--s1);color:#fff;border-color:var(--s1)" }, "FUSED")),
    h("div", { style: "display:flex;gap:6px;align-items:center;flex-wrap:wrap" },
      ...demandBadges(r, hasOracle), h("span", { class: "badge" }, `${r.latency_ms} ms`))));
  card.append(h("p", { class: "muted small", style: "margin:4px 0 12px" }, info.what || ""));
  if (!r.items.length) {
    card.append(h("p", { class: "muted small" }, "No dishes available for this request."));
    return card;
  }
  // Wider hero: show the 5 in two columns on a roomy screen.
  const wrap = h("div", { style: "columns:2;column-gap:28px" });
  wrap.append(dishList(r.items, colorOf));
  card.append(wrap);
  return card;
}

function algoCard(algo, r, hasOracle, colorOf) {
  const info = ALGORITHMS[algo] || {};
  const card = h("div", { class: "card" },
    h("div", { style: "display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap" },
      h("h2", { style: "margin:0" }, algoName(algo)),
      h("div", { style: "display:flex;gap:6px;align-items:center;flex-wrap:wrap" },
        ...demandBadges(r, hasOracle), h("span", { class: "badge" }, `${r.latency_ms} ms`))),
    h("p", { class: "muted small", style: "margin:4px 0 10px" }, info.what || ""));
  if (!r.items.length) {
    card.append(h("p", { class: "muted small" }, r.no_signal
      ? "No signal: this method needs the user's order history, and there is none here (cold start)."
      : "No dishes available for this request."));
    return card;
  }
  card.append(dishList(r.items, colorOf));
  return card;
}
