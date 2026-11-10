import { api, getMeta, h, fmt, withStates } from "../app.js";
import { bars } from "../charts.js";

const PIPELINE_SVG = `
<svg viewBox="0 0 900 150" role="img" aria-label="Two-stage recommendation pipeline">
  <style>.bx{fill:var(--surface-2);stroke:var(--border)}.t{fill:var(--text);font-size:13px;font-weight:600}.st{fill:var(--text-2);font-size:11px}.ar{stroke:var(--muted);stroke-width:1.5;fill:none;marker-end:url(#a)}</style>
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--muted)"/></marker></defs>
  <rect class="bx" x="5" y="35" width="150" height="80" rx="10"/><text class="t" x="20" y="62">Request</text><text class="st" x="20" y="82">user, location, time,</text><text class="st" x="20" y="98">weather</text>
  <rect class="bx" x="190" y="35" width="150" height="80" rx="10"/><text class="t" x="205" y="62">Availability</text><text class="st" x="205" y="82">open now and delivers</text><text class="st" x="205" y="98">here (PostGIS)</text>
  <rect class="bx" x="375" y="10" width="190" height="130" rx="10"/><text class="t" x="390" y="36">1 · Candidates (~100)</text><text class="st" x="390" y="58">buy-again, popular nearby,</text><text class="st" x="390" y="74">item-to-item, EASE / ALS,</text><text class="st" x="390" y="90">text embeddings, two-tower</text><text class="st" x="390" y="106">→ vector search (pgvector)</text><text class="st" x="390" y="126">fast, recall matters</text>
  <rect class="bx" x="600" y="10" width="160" height="130" rx="10"/><text class="t" x="615" y="36">2 · Ranking</text><text class="st" x="615" y="58">~40 features per pair</text><text class="st" x="615" y="74">LightGBM (3 losses)</text><text class="st" x="615" y="90">or DCNv2</text><text class="st" x="615" y="126">slow, precision matters</text>
  <rect class="bx" x="795" y="35" width="100" height="80" rx="10"/><text class="t" x="810" y="62">Top 5</text><text class="st" x="810" y="82">rules: ≤2 per</text><text class="st" x="810" y="98">restaurant</text>
  <path class="ar" d="M155,75H188"/><path class="ar" d="M340,75H373"/><path class="ar" d="M565,75H598"/><path class="ar" d="M760,75H793"/>
</svg>`;

export async function render(root, { signal }) {
  root.append(h("div", { class: "page-head" },
    h("h1", {}, "Food delivery recommendations: a lab"),
    h("p", { class: "lead" },
      "Synthetic but realistic food-delivery data for Barcelona and Madrid: restaurants, menus, users and six months of " +
      "sessions, feed impressions and orders. The lab trains several recommendation algorithms on it and compares their " +
      "quality, how differently they recommend, and what they cost to train and serve. The task: show each user the 5 dishes " +
      "that fit on one phone screen.")));
  const body = h("div");
  root.append(body);
  await withStates(body, async () => Promise.all([api("/api/overview", { signal }), getMeta()]), ([d, meta]) => draw(body, d, meta));
}

function draw(root, d, meta) {
  const c = d.counts;
  const tile = (v, k) => h("div", { class: "card tile" }, h("div", { class: "v" }, v), h("div", { class: "k" }, k));
  root.append(h("div", { class: "tiles" },
    tile(fmt.int(c.restaurants), "restaurants"), tile(fmt.int(c.items), "dishes and drinks"),
    tile(fmt.int(c.users), "users"), tile(fmt.int(c.sessions), "app sessions"),
    tile(fmt.int(c.orders), "orders"), tile(fmt.pct(c.conversion, 0), "sessions that order"),
    tile(fmt.eur(c.avg_basket_eur), "average basket")));

  const pipe = h("div", { class: "card" }, h("h2", {}, "How a recommendation is made"));
  const holder = h("div"); holder.innerHTML = PIPELINE_SVG;   // static markup, no user data
  pipe.append(holder, h("p", { class: "muted small" },
    "Stage 1 narrows thousands of dishes to about a hundred cheaply; stage 2 scores those carefully. Users with no history " +
    "get candidates from what is popular nearby at this hour; the ranker then uses context (distance, time, weather, price)."));
  root.append(pipe);

  const months = [...new Set(d.monthly.map((r) => r.month))];
  const cityOrders = (code) => months.map((m) => d.monthly.find((r) => r.month === m && r.city === code)?.orders || 0);
  const g1 = h("div", { class: "grid g2", style: "margin-top:16px" });
  const monthly = h("div", { class: "card" }, h("h2", {}, "Orders per month"), h("div"));
  const hourly = h("div", { class: "card" }, h("h2", {}, "Orders by hour of day"), h("div"));
  g1.append(monthly, hourly);
  root.append(g1);
  bars(monthly.lastChild, { categories: months, series: [
    { name: "Barcelona", values: cityOrders("BCN"), color: "var(--s1)" },
    { name: "Madrid", values: cityOrders("MAD"), color: "var(--s2)" }], format: fmt.int });
  const hours = [...Array(24).keys()];
  bars(hourly.lastChild, { categories: hours.map((x) => `${x}h`),
    series: [{ name: "Orders", values: hours.map((x) => d.hourly.find((r) => r.hour === x)?.orders || 0), color: "var(--s1)" }],
    format: fmt.int });
  monthly.append(h("p", { class: "muted small" }, "August dips (holidays); October grows as new users sign up. Train = May–Aug, validation = Sep, test = Oct."));
  hourly.append(h("p", { class: "muted small" }, "Spanish meal times: lunch peaks at 14h, dinner at 21h."));

  const label = Object.fromEntries(meta.cuisines.map((x) => [x.key, x.label]));
  const g2 = h("div", { class: "grid g2", style: "margin-top:16px" });
  const cu = h("div", { class: "card" }, h("h2", {}, "Orders by cuisine"), h("div"));
  const hist = h("div", { class: "card" }, h("h2", {}, "Users by order history before the test month"), h("div"));
  g2.append(cu, hist);
  root.append(g2);
  bars(cu.lastChild, { horizontal: true, categories: d.cuisine.map((r) => label[r.cuisine]),
    series: [{ name: "Orders", values: d.cuisine.map((r) => r.orders), color: "var(--s1)" }], format: fmt.int });
  const order = ["no history", "1-3 orders", "4-15 orders", "16+ orders"];
  bars(hist.lastChild, { categories: order,
    series: [{ name: "Users", values: order.map((b) => d.history_before_test.find((r) => r.bucket === b)?.users || 0), color: "var(--s1)" }],
    format: fmt.int });
  hist.append(h("p", { class: "muted small" },
    "Users with no history are the cold-start case: new sign-ups in October and people who never ordered. " +
    "They get recommendations from location, time and their sign-up profile only."));
  root.append(h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "How the data was made"),
    h("p", { class: "lead" },
      "Each simulated user has a hidden taste: favourite cuisines, a personal taste for individual dishes, price and distance " +
      "sensitivity, diet, spice tolerance, how strongly they reorder, and when they eat. In every session the app shows a feed; " +
      "the user orders (or not) by weighing taste, price, distance, delivery fee, time of day, weather and habit. About 10% of users " +
      "travel to the other city. Because the hidden taste is known, the lab can also score every recommendation by the true chance " +
      "the user would order it, which real data never allows."),
    h("p", { class: "note" }, "Results show how algorithms behave under this simulator's assumptions; they are not a forecast of production results.")));
}
