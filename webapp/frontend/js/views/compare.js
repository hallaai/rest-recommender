import { api, h, fmt, withStates } from "../app.js";
import { bars, heatmap, lines, showTip, hideTip } from "../charts.js";
import { ALGORITHMS, algoName } from "../algorithms.js";

const METRIC_LABELS = {
  demand: "True demand captured@5", hit: "HitRate@5", ndcg: "NDCG@5", rr: "MRR",
  recall100: "Recall@100", rest_hit: "Restaurant hit@5",
};

// Plain-language glossary for the metric headers. Each entry is the lines passed to showTip.
const METRIC_HELP = {
  demand: [
    "Demand@5 — the headline metric",
    "The expected share of demand the 5 shown dishes capture:",
    "the user's true probability of ordering each of the 5, summed.",
    "0.40 = the top 5 win ~40% of the realistically winnable orders.",
    "Not precision/recall. Because this is a simulation, the true order",
    "probability of EVERY dish is known, so this uses all of them —",
    "far less noisy than hit-rate, which sees one observed order.",
    "Closest to a probability-weighted precision@5.",
  ],
  hit: [
    "Hit@5 (HitRate@5)",
    "Did the one dish the user actually ordered appear in the 5 shown?",
    "A 0/1 per request, averaged. Standard, but noisy: it ignores the",
    "other 4 picks and how likely each was.",
  ],
  ndcg: [
    "NDCG@5",
    "Normalised Discounted Cumulative Gain. Rewards putting the",
    "right dish higher in the list (rank 1 counts more than rank 5).",
    "Measures ordering quality, not just presence.",
  ],
  rr: [
    "MRR (Mean Reciprocal Rank)",
    "1 / (rank of the ordered dish): 1.0 if it's first, 0.5 if second…",
    "0 if it's not in the 5. Rewards getting the best pick to the top.",
  ],
  recall100: [
    "Recall@100",
    "Did the ordered dish survive into the ~100 pooled candidates,",
    "before ranking? Measures the retrieval stage on its own —",
    "the ranker can only work with what retrieval hands it.",
  ],
  rest_hit: [
    "Restaurant hit@5",
    "Was the ordered dish's RESTAURANT represented in the 5 shown?",
    "Looser than Hit@5 — credits recommending the right place even",
    "if not the exact dish.",
  ],
  pct_best: [
    "% of best",
    "Demand@5 divided by the best achievable Demand@5 for each",
    "request (the 5 highest-probability dishes). 73% = the model",
    "captured 73% of the ceiling a perfect ranker could reach.",
  ],
  cold_demand: [
    "Cold-start demand",
    "Demand@5 measured only on requests from users with no order",
    "history. The history-based methods score ~0 here; this is where",
    "the pipeline earns its cost.",
  ],
  train: [
    "Train",
    "Wall-clock seconds to train this ranker on the September data.",
    "An offline batch cost, not part of serving latency.",
  ],
};

// A table header cell carrying a plain-language tooltip (ⓘ cue, hover to read).
function metricTh(label, key, numeric = true) {
  const lines = METRIC_HELP[key];
  const th = h("th", { class: numeric ? "num" : "" },
    label, lines ? h("span", { class: "muted", style: "margin-left:4px" }, "ⓘ") : null);
  if (lines) {
    th.style.cursor = "help";
    th.addEventListener("mousemove", (e) => showTip(e, lines));
    th.addEventListener("mouseleave", hideTip);
  }
  return th;
}

const NAMES = Object.fromEntries(Object.entries(ALGORITHMS).map(([k, v]) => [k, v.name]));

export async function render(root, { signal }) {
  root.append(h("div", { class: "page-head" }, h("h1", {}, "Compare"),
    h("p", { class: "lead" }, "How the retrieval baselines score on the October test month, how differently they " +
      "recommend, and what they cost. Bars show 95% confidence intervals (bootstrap over users).")));
  const body = h("div");
  root.append(body);
  await withStates(body, () => api("/api/leaderboard", { signal }), (d) => draw(body, d));
  // The two-stage pipeline section loads separately; absent until rank_runner has run.
  const pipe = h("div");
  root.append(pipe);
  try {
    const r = await api("/api/ranking", { signal });
    drawPipeline(pipe, r);
  } catch { /* no ranking.json yet: section stays empty */ }

  // Phase-5 sections: each loads independently and stays absent until its artifact exists.
  const extra = h("div");
  root.append(extra);
  let anyExtra = false;
  const header = h("h1", { style: "margin-top:28px" }, "Deeper experiments (phase 5)");
  for (const [path, drawer] of [
    ["/api/ann", drawAnn], ["/api/scaling", drawScaling],
    ["/api/bandit", drawBandit], ["/api/gemini_rerank", drawRerank],
  ]) {
    try {
      const d = await api(path, { signal });
      if (!anyExtra) { extra.append(header); anyExtra = true; }
      drawer(extra, d);
    } catch { /* artifact not generated yet */ }
  }
}

function sectionCard(root, title, lead) {
  const card = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, title));
  if (lead) card.append(h("p", { class: "muted small", style: "margin-top:0" }, lead));
  root.append(card);
  return card;
}

function drawAnn(root, d) {
  const card = sectionCard(root, "Approximate vector search (ANN)",
    `Does an approximate index cost us quality? ${d.n_queries} real taste-vector queries over ${fmt.int(d.n_items)} ` +
    `dishes (${d.dim}-d ${d.backend}), recall@${d.k} measured against exact brute-force truth. ` +
    "Latency is dominated by the round-trip to the managed database, so read the differences, not the absolute ms.");
  card.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Strategy", "Setting", `Recall@${d.k}`, "Median ms", "p95 ms"].map((x, i) =>
      h("th", { class: i > 1 ? "num" : "" }, x)))),
    h("tbody", {}, d.strategies.map((s) => h("tr", {},
      h("td", {}, s.name), h("td", { class: "muted small" }, s.param || "—"),
      h("td", { class: "num" }, s.recall.toFixed(3)),
      h("td", { class: "num" }, s.median_ms.toFixed(1)),
      h("td", { class: "num" }, s.p95_ms.toFixed(1))))))));
  card.append(h("p", { class: "muted small" }, "Takeaway: HNSW reaches near-exact recall (raise ef_search to trade " +
    "latency for recall); IVFFlat trades recall for speed via probes. Exact scan is the correctness reference — at this " +
    "catalogue size it is already fast, so an index earns its keep mainly as the catalogue grows (see scaling)."));
}

function drawScaling(root, d) {
  const m = d.measured;
  const card = sectionCard(root, "Scaling projection",
    `Measured at ${fmt.int(m.n_items)} dishes / ${fmt.int(m.n_users)} users, projected along each stage's complexity. ` +
    "Serving latency grows only with log(items); ranker training grows with users (a scheduled batch job, off the serving path).");
  card.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Scenario", "Dishes", "Users", "Ranker train", "Serve / req", "Vectors (×3 spaces)"].map((x, i) =>
      h("th", { class: i ? "num" : "" }, x)))),
    h("tbody", {}, d.scenarios.map((s) => h("tr", {},
      h("td", {}, s.scenario),
      h("td", { class: "num" }, fmt.int(s.items)),
      h("td", { class: "num" }, fmt.int(s.users)),
      h("td", { class: "num" }, s.train_seconds != null ? `${s.train_seconds.toFixed(0)} s` : "–"),
      h("td", { class: "num" }, s.serve_ms_per_request != null ? `${s.serve_ms_per_request.toFixed(1)} ms` : "–"),
      h("td", { class: "num" }, `${(s.vector_mb_per_space * 3).toFixed(0)} MB`)))))));
  card.append(h("ul", { class: "muted small" }, d.notes.map((n) => h("li", {}, n))));
}

function drawBandit(root, d) {
  const card = sectionCard(root, "Cold-start bandit vs static policy",
    `Thompson-sampling over cuisine arms for the ${fmt.int(d.n_requests)} no-history requests, shared across ` +
    `${d.n_segments} (city, age) segments. Reward is true captured demand. The static line is the pipeline's own ` +
    "cold-start policy (geo + time popularity) — a strong prior the bandit must learn to beat.");
  const chart = h("div");
  card.append(chart);
  lines(chart, {
    x: d.curve.map((p) => String(p.round)), height: 240, format: (v) => v.toFixed(3),
    series: [
      { name: "Bandit (cumulative demand)", values: d.curve.map((p) => p.bandit), color: "var(--s1)" },
      { name: "Static geo+time popularity", values: d.curve.map((p) => p.static), color: "var(--s3)" },
    ],
  });
  const verdict = d.final_bandit_demand >= d.final_static_demand
    ? `The bandit overtakes the static prior (${d.final_bandit_demand.toFixed(3)} vs ${d.final_static_demand.toFixed(3)}).`
    : `The bandit's curve climbs as it learns but does not overtake the static prior here ` +
      `(${d.final_bandit_demand.toFixed(3)} vs ${d.final_static_demand.toFixed(3)}). Geo+time popularity is already a ` +
      "very strong cold-start prior; exploration earns its keep only if the cold segment is large and revisited often.";
  card.append(h("p", { class: "muted small" }, verdict));
}

function drawRerank(root, d) {
  const lift = d.lift_pct;
  const card = sectionCard(root, "Gemini LLM reranker",
    `${d.model} reranks the pipeline's top ${d.rerank_n} candidates using dish text + context, on ${fmt.int(d.n_requests)} ` +
    "requests. Same candidate set as LightGBM, so this isolates the rerank step. Reward is true captured demand.");
  card.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {},
      h("th", {}, "Policy"), metricTh("Demand@5", "demand"),
      h("th", { class: "num" }, "Median latency"), h("th", { class: "num" }, "Spend"))),
    h("tbody", {}, [
      h("tr", {}, h("td", {}, "LightGBM listwise"), h("td", { class: "num" }, (d.lgbm_demand ?? 0).toFixed(3)),
        h("td", { class: "num" }, "~10 ms"), h("td", { class: "num" }, "$0")),
      h("tr", {}, h("td", {}, "+ Gemini rerank"), h("td", { class: "num" }, (d.gemini_demand ?? 0).toFixed(3)),
        h("td", { class: "num" }, d.median_llm_latency_ms != null ? `${d.median_llm_latency_ms.toFixed(0)} ms` : "–"),
        h("td", { class: "num" }, `$${(d.spend_usd ?? 0).toFixed(4)}`)),
    ]))));
  const verdict = lift == null ? "No lift computed."
    : `Lift from the LLM rerank: ${lift >= 0 ? "+" : ""}${lift.toFixed(1)}%. ` +
      (Math.abs(lift) < 2
        ? "Within noise — the LightGBM ranker already orders these candidates well, and an LLM on the serving path is "
          + `~${d.median_llm_latency_ms != null ? Math.round(d.median_llm_latency_ms) : "100s of"} ms and paid per request.`
        : "A measurable change, but weigh it against the per-request latency and dollar cost of an LLM in the path.") +
      (d.llm_malformed ? ` (${d.llm_malformed} replies were malformed and fell back to the LightGBM order.)` : "") +
      (d.stopped ? ` Run stopped early: ${d.stopped}.` : "");
  card.append(h("p", { class: "muted small" }, verdict));
}

function drawPipeline(root, r) {
  root.append(h("h1", { style: "margin-top:28px" }, "Two-stage pipeline (phase 3)"),
    h("p", { class: "lead" }, "The full design: candidates pooled from the retrieval sources, then re-ranked by a " +
      "model using ~40 features per (user, dish) pair — distance, time, price, history and each retrieval score. " +
      `Trained on September (${fmt.int(r.n_train_rows)} rows), tested on October (${fmt.int(r.n_test)} requests).`));
  const pipes = [...r.pipelines].sort((a, b) => b.metrics.demand.mean - a.metrics.demand.mean);
  const card = h("div", { class: "card" }, h("h2", {}, "Ranker comparison"));
  card.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {},
      h("th", {}, "Ranker (loss)"),
      metricTh("Demand@5", "demand"), metricTh("Hit@5", "hit"), metricTh("NDCG@5", "ndcg"),
      metricTh("% of best", "pct_best"), metricTh("Cold-start demand", "cold_demand"),
      metricTh("Train", "train"))),
    h("tbody", {}, pipes.map((p) => h("tr", {},
      h("td", {}, rankerLabel(p.ranker)),
      h("td", { class: "num" }, p.metrics.demand.mean.toFixed(3)),
      h("td", { class: "num" }, p.metrics.hit.mean.toFixed(3)),
      h("td", { class: "num" }, p.metrics.ndcg.mean.toFixed(3)),
      h("td", { class: "num" }, fmt.pct(p.pct_of_best, 0)),
      h("td", { class: "num" }, p.cold_demand.mean.toFixed(3)),
      h("td", { class: "num" }, `${p.train_seconds.toFixed(1)} s`)))))));
  card.append(h("p", { class: "muted small" }, "The three LightGBM losses: pointwise (logloss), pairwise (LambdaRank), " +
    "listwise (rank_xendcg, tuned for the top of the list). DCNv2 is the neural challenger. Compare the best pipeline " +
    "here against the best single retriever above — the fusion should win, and lift cold-start most."));
  root.append(card);

  // Feature importance of the best pipeline
  const best = pipes.find((p) => p.feature_importance) || pipes[0];
  if (best?.feature_importance) {
    const imp = Object.entries(best.feature_importance).sort((a, b) => b[1] - a[1]).slice(0, 15);
    const fc = h("div", { class: "card", style: "margin-top:16px" },
      h("h2", {}, `What the ranker relies on (${rankerLabel(best.ranker)})`), h("div"));
    root.append(fc);
    bars(fc.lastChild, { horizontal: true, labelWidth: 170, categories: imp.map((x) => prettyFeat(x[0])),
      series: [{ name: "Importance", values: imp.map((x) => x[1]), color: "var(--s3)" }], format: fmt.int });
    fc.append(h("p", { class: "muted small" }, "LightGBM split counts: how often each feature is used to split a tree. " +
      "Note the retrieval scores and the hard features (distance, price, recency) sit side by side — that is the fusion."));
  }
}

function rankerLabel(r) {
  return { pointwise: "LightGBM pointwise", pairwise: "LightGBM pairwise", listwise: "LightGBM listwise", dcnv2: "DCNv2 (neural)" }[r] || r;
}
function prettyFeat(f) {
  return f.replace(/_/g, " ").replace("retr ", "score: ");
}

function nm(k) { return NAMES[k] || k; }

function ciCell(ci) {
  if (!ci) return h("td", { class: "num" }, "–");
  const w = ci.hi != null ? ` ±${(((ci.hi - ci.lo) / 2) * 100).toFixed(1)}` : "";
  return h("td", { class: "num", title: ci.hi != null ? `95% CI [${ci.lo.toFixed(3)}, ${ci.hi.toFixed(3)}]` : "" },
    (ci.mean).toFixed(3), w && h("span", { class: "muted small" }, w));
}

function draw(root, d) {
  const algos = [...d.algorithms].sort((a, b) => b.metrics.demand.mean - a.metrics.demand.mean);
  root.append(h("p", { class: "note" }, `${fmt.int(d.n_requests)} test requests (October)` +
    (d.sample ? `, sampled` : "") + `. Best by demand: ${nm(d.best_by_demand)}. ` +
    "The headline metric, Demand@5, is the expected share of demand the 5 shown dishes capture — the user's true " +
    "order-probability for each of the 5, summed, from the simulation's hidden choice model. It is not precision or " +
    "recall: it uses every dish's probability (not one observed order), so it is far less noisy. Hover any column header " +
    "for its definition."));

  // Leaderboard table
  const metricCols = ["demand", "hit", "ndcg", "rr", "recall100", "rest_hit"];
  const card = h("div", { class: "card" }, h("h2", {}, "Leaderboard"));
  card.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {},
      h("th", {}, "Algorithm"),
      metricCols.map((m) => metricTh(shortLabel(m), m)),
      metricTh("% of best", "pct_best"),
      h("th", { class: "num" }, "€/1k sessions"))),
    h("tbody", {}, algos.flatMap((a) => {
      const info = ALGORITHMS[a.name];
      return [
        h("tr", {},
          h("td", {}, nm(a.name), a.name === d.best_by_demand ? [" ", h("span", { class: "badge" }, "best")] : null),
          metricCols.map((m) => ciCell(a.metrics[m])),
          h("td", { class: "num" }, fmt.pct(a.metrics.pct_of_best.mean, 0)),
          h("td", { class: "num" }, fmt.eur(a.metrics.revenue_per_1000.mean))),
        info && h("tr", {}, h("td", { colspan: 9, class: "muted small", style: "padding-top:0;border-bottom:1px solid var(--border)" },
          `${info.stage} · ${info.what}`)),
      ].filter(Boolean);
    })))));
  card.append(h("p", { class: "muted small" }, "± is half the 95% interval width. Hover a cell for the exact interval."));
  root.append(card);

  // Demand with CI as a bar chart
  const demandCard = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "True demand captured@5"), h("div"));
  root.append(demandCard);
  bars(demandCard.lastChild, { horizontal: true, labelWidth: 170,
    categories: algos.map((a) => nm(a.name)),
    series: [{ name: "Demand", values: algos.map((a) => a.metrics.demand.mean), color: "var(--s1)" }],
    format: (v) => v.toFixed(3) });

  // Quality vs cost scatter (fit time)
  root.append(qualityVsCost(algos));

  // Overlap heatmap
  const names = d.overlap.names;
  const ov = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "How much algorithms overlap"),
    h("p", { class: "muted small" }, "Average Jaccard similarity of the top-5 lists for the same request (1 = identical, 0 = disjoint)."), h("div"));
  root.append(ov);
  heatmap(ov.lastChild, { rows: names.map(nm), cols: names.map(nm), values: d.overlap.jaccard,
    labelWidth: 170, format: (v) => v.toFixed(2) });

  // Segments: demand by user history
  root.append(segmentCard(algos, "history", "Demand by user history (orders before the test month)",
    ["none", "1-3", "4-15", "16+"]));
  root.append(segmentCard(algos, "traveller", "Demand: home city vs the other city", ["no", "yes"],
    { no: "home city", yes: "other city" }));
}

function shortLabel(m) {
  return { demand: "Demand@5", hit: "Hit@5", ndcg: "NDCG@5", rr: "MRR", recall100: "Recall@100", rest_hit: "Rest@5" }[m];
}

function qualityVsCost(algos) {
  const card = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Quality vs training cost"),
    h("p", { class: "muted small" }, "Demand captured against fit time (log scale). Up and to the left is better."));
  const W = 640, H = 300, pl = 48, pb = 36, plotW = W - pl - 12, plotH = H - pb - 10;
  const maxD = Math.max(...algos.map((a) => a.metrics.demand.mean)) * 1.1;
  const times = algos.map((a) => Math.max(a.cost.fit_s, 0.01));
  const lx = (t) => Math.log10(t), minL = Math.min(...times.map(lx)) - 0.3, maxL = Math.max(...times.map(lx)) + 0.3;
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "");
  const el = (t, a) => { const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); return n; };
  const X = (t) => pl + ((lx(t) - minL) / (maxL - minL)) * plotW;
  const Y = (v) => 10 + plotH - (v / maxD) * plotH;
  for (let g = 0; g <= 4; g++) { const y = 10 + plotH - (g / 4) * plotH;
    svg.append(el("line", { x1: pl, x2: W, y1: y, y2: y, stroke: "var(--grid)" }));
    const tx = el("text", { x: pl - 6, y: y + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11 }); tx.textContent = (maxD * g / 4).toFixed(2); svg.append(tx); }
  for (const t of [0.01, 0.1, 1, 10, 100]) { if (lx(t) < minL || lx(t) > maxL) continue;
    const tx = el("text", { x: X(t), y: H - 10, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); tx.textContent = t < 1 ? `${t}s` : `${t}s`; svg.append(tx); }
  // Nudge labels apart when two dots are close vertically, and flip left near the right edge.
  const placed = [];
  algos.forEach((a, i) => {
    const color = `var(--s${(i % 8) + 1})`;
    const cx = X(Math.max(a.cost.fit_s, 0.01)), cy = Y(a.metrics.demand.mean);
    svg.append(el("circle", { cx, cy, r: 6, fill: color, stroke: "var(--surface)", "stroke-width": 2 }));
    const right = cx > W * 0.7;
    let ly = cy + 4;
    while (placed.some((p) => Math.abs(p - ly) < 13)) ly += 13;          // avoid overlap
    placed.push(ly);
    const lbl = el("text", { x: cx + (right ? -10 : 10), y: ly, "text-anchor": right ? "end" : "start",
      fill: "var(--text-2)", "font-size": 11 });
    lbl.textContent = nm(a.name); svg.append(lbl);
  });
  const wrap = h("div", { class: "chart" }); wrap.append(svg); card.append(wrap);
  return card;
}

function segmentCard(algos, key, title, order, relabel = {}) {
  const card = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, title), h("div"));
  const cats = order.map((o) => relabel[o] || o);
  const series = algos.slice(0, 5).map((a, i) => ({
    name: nm(a.name), color: `var(--s${(i % 8) + 1})`,
    values: order.map((o) => a.segments[key]?.[o]?.demand?.mean ?? 0),
  }));
  bars(card.lastChild, { categories: cats, series, format: (v) => v.toFixed(3), height: 240 });
  card.append(h("p", { class: "muted small" }, "Top 5 algorithms by overall demand. Cold users (no history) are the hardest case."));
  return card;
}
