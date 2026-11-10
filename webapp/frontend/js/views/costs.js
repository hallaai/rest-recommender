import { api, h, fmt, withStates } from "../app.js";
import { bars } from "../charts.js";

const NAMES = {
  random: "Random", popularity: "Popularity", geo_time_pop: "Geo + time popularity",
  buy_again: "Buy-again", item_item: "Item-to-item", ease: "EASE", als: "ALS matrix factorisation",
};
const nm = (k) => NAMES[k] || k;

export async function render(root, { signal }) {
  root.append(h("div", { class: "page-head" }, h("h1", {}, "Costs"),
    h("p", { class: "lead" }, "What each algorithm costs to train and to serve: fit time and memory, latency per " +
      "recommendation, and an estimated compute price. Measured on the machine that ran the evaluation (12 CPU cores, no GPU).")));
  const body = h("div");
  root.append(body);
  await withStates(body, () => api("/api/leaderboard", { signal }), (d) => draw(body, d));
}

function draw(root, d) {
  const algos = [...d.algorithms].sort((a, b) => a.cost.fit_s - b.cost.fit_s);
  const card = h("div", { class: "card" }, h("h2", {}, "Training and serving cost"));
  card.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Algorithm", "Fit time", "Peak memory", "Latency p50", "Latency p95", "€ / million recs"].map((x, i) =>
      h("th", { class: i ? "num" : "" }, x)))),
    h("tbody", {}, algos.map((a) => h("tr", {},
      h("td", {}, nm(a.name)),
      h("td", { class: "num" }, `${a.cost.fit_s.toFixed(1)} s`),
      h("td", { class: "num" }, `${a.cost.peak_mb.toFixed(0)} MB`),
      h("td", { class: "num" }, `${a.cost.lat_p50_ms.toFixed(2)} ms`),
      h("td", { class: "num" }, `${a.cost.lat_p95_ms.toFixed(2)} ms`),
      h("td", { class: "num" }, a.cost.usd_per_million_recs < 0.01 ? "<€0.01" : fmt.eur(a.cost.usd_per_million_recs))))))));
  card.append(h("p", { class: "muted small" }, "“€ / million recs” uses only the ranking time at the reference vCPU-hour price; " +
    "it excludes candidate retrieval, which the two-stage pipeline (phase 3) adds."));
  root.append(card);

  const fitCard = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Fit time (seconds)"), h("div"));
  root.append(fitCard);
  bars(fitCard.lastChild, { horizontal: true, labelWidth: 170, categories: algos.map((a) => nm(a.name)),
    series: [{ name: "Fit seconds", values: algos.map((a) => a.cost.fit_s), color: "var(--s2)" }],
    format: (v) => v.toFixed(1) });

  const latCard = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Latency per recommendation (p95, ms)"), h("div"));
  root.append(latCard);
  const byLat = [...algos].sort((a, b) => a.cost.lat_p95_ms - b.cost.lat_p95_ms);
  bars(latCard.lastChild, { horizontal: true, labelWidth: 170, categories: byLat.map((a) => nm(a.name)),
    series: [{ name: "p95 ms", values: byLat.map((a) => a.cost.lat_p95_ms), color: "var(--s4)" }],
    format: (v) => v.toFixed(2) });

  root.append(h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Reading these costs"),
    h("p", { class: "lead" }, "Fit happens once per retraining (nightly or weekly); latency happens on every request. " +
      "A cheap-to-serve model with a slow fit (ALS) is fine for production; a slow-per-request model (item-to-item here) " +
      "is the opposite. The two-stage pipeline in phase 3 adds candidate-retrieval cost (pgvector lookups) on top of ranking.")));
}
