import { h } from "../app.js";
import { ALGORITHMS } from "../algorithms.js";

const card = (title, ...body) => h("div", { class: "card", style: "margin-bottom:16px" }, h("h2", {}, title), ...body);
const p = (t) => h("p", { class: "lead" }, t);
const table = (head, rows) => h("div", { class: "table-wrap" }, h("table", {},
  h("thead", {}, h("tr", {}, head.map((x) => h("th", {}, x)))),
  h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c)))))));

function architectureCard() {
  const c = card("Two-stage architecture: how features are used");
  c.append(p("A recommendation is not made by one model. If it went only by dish-description similarity it " +
    "would ignore distance, price, time, the user's history, popularity, and whether the restaurant is even " +
    "open. Instead two stages run in sequence, and text-embedding similarity is just one signal among several."));
  c.append(h("h3", {}, "Stage 1 — Candidates (about 100 dishes)"),
    p("Several retrieval sources each propose dishes. They are cheap and favour recall (don't miss a good dish). " +
      "Text-embedding similarity (phase 3) is one of them, not the whole stage:"),
    table(["Source", "Signal it uses", "Suggests new dishes?"],
      ["geo_time_pop", "popularity", "buy_again", "item_item", "ease", "als", "content_ann", "two_tower"].map((k) => {
        const a = ALGORITHMS[k];
        return [a.name, a.signal, a.news];
      })));
  c.append(h("h3", { style: "margin-top:14px" }, "Stage 2 — Ranking"),
    p("The pooled ~100 candidates are scored by a ranking model (LightGBM, then DCNv2) using about 40 features " +
      "per (user, dish) pair. Embedding similarity is only a few of those columns:"),
    h("ul", { class: "lead" },
      ["distance to the restaurant, delivery fee, prep time",
       "time fit: does this cuisine match the user's usual hour and weekday?",
       "price fit: dish price vs the user's average basket",
       "history counts: times this cuisine / restaurant ordered in the last 7 / 30 / 90 days; days since last",
       "popularity of the dish in this zone right now",
       "each retrieval source's score, including the embedding cosine similarity",
       "restaurant rating, is-reorder flag, diet match"].map((t) => h("li", {}, t))),
    h("p", { class: "note" }, "So ANN never decides the final list alone. A dish with a perfect description match but " +
      "8 km away at a closed restaurant is filtered out (availability) or ranked down (distance feature). " +
      "Stage 1 and the ranker are built in phase 3; phase 2 measures the retrieval sources on their own."));
  return c;
}

function streamsCard() {
  const c = card("From two retrieval streams to one screen of 5");
  c.append(p("Candidates come from two kinds of stream that answer different questions, then one ranker fuses " +
    "them into the five dishes on screen. Here is the whole path, and what each step contributes to the chance " +
    "the user actually orders."));

  c.append(h("h3", {}, "Stream A — vector (embedding / ANN) retrieval"),
    p("Each dish is turned into a 768-dimension vector from its text (name, description, cuisine, diet, restaurant). " +
      "The user's vector is the time-weighted average of the dishes they have ordered (recent orders weigh more). " +
      "pgvector returns the nearest dishes by cosine similarity (HNSW index). This stream works on meaning, so it " +
      "surfaces dishes the user never ordered and carries a traveller's taste across cities. Two-tower is a second " +
      "vector stream whose user vector also reads profile features, so it still has signal for a brand-new user."),
    table(["Vector source", "User vector is…", "Best demand@5 on its own"], [
      ["content (Gemini embeddings)", "avg of ordered-dish vectors", "0.231"],
      ["content (BGE / TF-IDF)", "same, cheaper backends", "0.224 / 0.216"],
      ["two-tower network", "learned from profile + history", "0.179 (works cold)"]]));

  c.append(h("h3", { style: "margin-top:14px" }, "Stream B — non-vector retrieval"),
    p("These never look at dish text. They read the order matrix (who ordered what) and simple counts, and they are " +
      "the strongest single signals for users who already have history:"),
    table(["Non-vector source", "Signal", "Demand@5 on its own"], [
      ["buy-again", "your own past orders (recency-weighted)", "0.274 — best single"],
      ["ALS matrix factorisation", "collaborative filtering", "0.200"],
      ["item-to-item", "dishes co-ordered over time", "0.153"],
      ["EASE", "closed-form item-item CF", "0.097"],
      ["geo + time popularity", "location + hour, no history needed", "0.065 (works cold)"]]));

  c.append(h("h3", { style: "margin-top:14px" }, "Pooling — one candidate set of ~100"),
    p("Each stream proposes its top dishes; the union is blended by rank (each source contributes a 0–1 rank score) " +
      "and capped at ~100 candidates. Only dishes from restaurants open now and delivering to the user are eligible. " +
      "No stream decides the final list — pooling just guarantees a good dish is not missed before ranking."));

  c.append(h("h3", { style: "margin-top:14px" }, "Ranking — the one reranker that fuses everything"),
    p("There is not a reranker per stream. The ~100 pooled candidates go to a single ranking model — LightGBM with a " +
      "listwise loss (rank_xendcg) is the one we serve; pairwise LambdaRank and a DCNv2 neural net were tested and do " +
      "not beat it. It scores each (user, dish) pair on ~35 features at once, so every stream becomes a few input columns:"),
    h("ul", { class: "lead" },
      [["Each stream's score", "the vector cosine and every non-vector score are columns the ranker weighs"],
       ["Hard features", "distance, delivery fee, prep time — a perfect text match 8 km away is ranked down"],
       ["Fit features", "price vs the user's average basket; cuisine vs their usual hour and weekday"],
       ["History features", "times this cuisine / restaurant ordered in 7 / 30 / 90 days; days since last; reorder flag"],
      ].map(([k, v]) => h("li", {}, h("b", {}, k + ": "), v))),
    p("The ranker's own top features are distance, the content (vector) score, geo-popularity, the ALS score and price — " +
      "vector and non-vector signals side by side. It sorts the pool, the variety rule keeps at most 2 dishes per " +
      "restaurant, and the top 5 are shown."));

  c.append(h("h3", { style: "margin-top:14px" }, "Probability to buy, stream by stream and fused"),
    p("“Demand@5” is the chance the user orders one of the 5 shown — the sum of their true order-probabilities for " +
      "those 5, from the simulation's hidden model. It lets us put a number on each stream and on the fusion:"),
    table(["If the 5 came only from…", "Chance user buys one of the 5"], [
      ["buy-again (best single stream)", "0.274"],
      ["best vector stream (Gemini content)", "0.231"],
      ["ALS", "0.200"],
      ["geo + time popularity (cold-start safe)", "0.065"],
      ["the FUSED ranker — what ships", "0.402  (+48% over the best single)"],
      ["— of which, new users with no history", "0.075  (every history stream ≈ 0 here)"]]),
    h("p", { class: "note" }, "The fusion is worth ~48% more captured demand than any single stream, and it is the only " +
      "thing that gives a cold-start user a real list. That gap is the whole argument for the two-stage design."));
  return c;
}

function conversionCard() {
  const c = card("Who buys, how much: the demand the feed is scored against");
  c.append(p("The five dishes are shown in a session. Not every session converts, and a buyer usually adds more than " +
    "the main dish. These are the real figures from the simulated six months, and they frame what “demand captured” means."));
  c.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Figure", "Value", "Meaning"].map((x) => h("th", {}, x)))),
    h("tbody", {}, [
      ["Session conversion", "62.1%", "95,681 orders from 154,005 sessions — chance a visit ends in an order"],
      ["Users who ever buy", "91.2%", "5,470 of 6,000 users place at least one order; 530 never order"],
      ["Orders per user (lifetime)", "15.9 avg", "17.5 per buyer, over the six months; some have 0, some 50+"],
      ["Items per order", "1.96 avg", "1 main dish + ~0.96 add-ons (sides, drinks, dessert) on average"],
      ["Users with no history (cold)", "25%", "1,505 of 6,000 have no orders before the test month — the hard case"],
    ].map(([a, b, d2]) => h("tr", {},
      h("td", {}, h("b", {}, a)), h("td", { class: "num" }, b), h("td", { class: "small" }, d2)))))));
  c.append(h("p", { class: "note" }, "So a single top-5 feed is one 62%-likely purchase of ~2 items. “Demand@5 = 0.40” " +
    "for the fused pipeline means that, averaged over all requests, the five shown capture about 40% of the order " +
    "probability the best possible five could — the lever the whole system pulls on."));
  return c;
}

function catalogCard() {
  const c = card("The algorithms");
  c.append(p("Every algorithm fits the same interface (fit on past orders, then score candidate dishes). " +
    "Phase 2 evaluates the retrieval baselines; the embedding retriever, two-tower and ranker arrive in phase 3 (greyed)."));
  const order = ["random", "popularity", "geo_time_pop", "buy_again", "item_item", "ease", "als",
                 "content_ann", "two_tower", "ranker"];
  const phase3 = new Set(["content_ann", "two_tower", "ranker"]);
  c.append(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["Algorithm", "Stage", "What it is", "New dishes?"].map((x) => h("th", {}, x)))),
    h("tbody", {}, order.map((k) => {
      const a = ALGORITHMS[k];
      return h("tr", { style: phase3.has(k) ? "opacity:.6" : "" },
        h("td", {}, a.name, phase3.has(k) ? [" ", h("span", { class: "badge" }, "phase 3")] : null),
        h("td", { class: "small" }, a.stage), h("td", { class: "small" }, a.what),
        h("td", { class: "small" }, a.news));
    })))));
  return c;
}

function top5Card() {
  const c = card("The top 5 on screen: how they are chosen");
  c.append(p("The five dishes are one ranked list, not fixed slots. Buy-again (reorders) and the discovery sources " +
    "(ALS, item-to-item, popularity, embeddings) all feed the same candidate pool; the ranker sorts the pool and the top " +
    "five are shown. So a typical list is a mix — some favourites the user reorders, some new dishes the model expects " +
    "them to like — in whatever order their scores fall."));
  c.append(h("ul", { class: "lead" },
    [["One ranked list", "no separate “old” and “new” slots; the mix is whatever scores highest."],
     ["Variety rule", "at most 2 dishes per restaurant in the five, so the screen is not five pizzas from one place."],
     ["Availability first", "only dishes from restaurants that are open now and deliver to the user are considered at all."],
    ].map(([k, v]) => h("li", {}, h("b", {}, k + ": "), v))));
  c.append(h("h3", { style: "margin-top:10px" }, "“Dishes usually bought together” is a different task"),
    p("Recommending a main dish (this system) is not the same as recommending what goes with it. A burger-and-fries pairing " +
      "is complementary / basket recommendation, and real apps handle it as a separate step after the user picks a main: " +
      "you add a burger to the basket, then the app suggests fries or a drink from that restaurant. In this lab those " +
      "add-ons are the sides, drinks and desserts attached to an order; a dedicated “frequently added” recommender on " +
      "dish-click is a possible later feature, kept separate from the top-5 main-dish feed."),
    h("p", { class: "note" }, "Item-to-item here measures substitute-style co-occurrence over time (people who order X also " +
      "order Y), which is about taste similarity — not “these two are eaten together in one meal”."));
  return c;
}

export function render(root) {
  root.append(h("div", { class: "page-head" }, h("h1", {}, "Method"),
    h("p", { class: "lead" }, "How the data is made, what is embedded, and how the algorithms are measured and compared.")));

  root.append(architectureCard());
  root.append(streamsCard());
  root.append(conversionCard());
  root.append(catalogCard());
  root.append(top5Card());

  root.append(card("The task and the time split",
    p("Each test case is a real order from October: we know who ordered, where, when and in what weather. The algorithm " +
      "sees every main dish it could recommend at that moment, from restaurants that are open and deliver there (about 1,200), " +
      "and ranks them. The dish the user actually ordered is the answer we check against."),
    table(["Period", "Months", "Used for"], [
      ["Train", "May–Aug", "Fitting the candidate models"],
      ["Validation", "Sep", "Choosing settings; training the ranker on candidates from models fitted on May–Aug"],
      ["Test", "Oct", "All reported numbers (models refitted on May–Sep)"]]),
    h("p", { class: "note" }, "No peeking at the future: an algorithm only sees orders from before the day of the request, like a nightly data refresh.")));

  root.append(card("Quality metrics (top 5 = one phone screen)",
    table(["Metric", "Plain meaning"], [
      ["HitRate@5", "How often the dish the user ordered is among the 5 shown"],
      ["NDCG@5", "Like HitRate, but a hit in 1st place counts more than in 5th"],
      ["MRR", "How high the ordered dish appears, on average (1 / position)"],
      ["Recall@100", "Candidate stage only: did the right dish make it into the 100 passed to the ranker"],
      ["Restaurant hit@5", "At least one of the 5 dishes is from the restaurant the user ordered from"],
      ["True demand captured@5", "From the simulation's hidden truth: the chance the user orders one of the 5 shown. Uses every dish's real probability, not just one observed order, so it is much less noisy"],
      ["% of best possible", "Captured demand compared with the best 5 dishes the hidden truth would pick"],
      ["Revenue per 1,000 sessions", "Captured demand × price, in EUR"]]),
    p("Behaviour, not just accuracy: share of reorders vs new dishes, how many restaurants and cuisines appear in the 5, " +
      "catalogue coverage, popularity bias, how evenly restaurants get exposure (Gini), and how much two algorithms' lists " +
      "overlap for the same request (Jaccard). Every number is also split by user history (none, 1–3, 4–15, 16+ orders), " +
      "travellers ordering in the other city, city and meal time.")));

  root.append(card("Confidence intervals",
    p("Every metric is shown with a 95% bootstrap interval: we resample users (with all their orders together, since one person's " +
      "orders are related) 1,000 times and take the middle 95% of the results. When two algorithms are compared, the difference " +
      "is resampled on the same users (paired bootstrap); a gap counts as real only if its interval does not include zero."),
    h("p", { class: "note" }, "Small groups have wide intervals: about 1,000 orders from users with no history and about 250 from travellers. Treat differences there with care.")));

  root.append(card("Cost metrics",
    p("Training time (wall and CPU), peak memory, model size, time per recommendation (median and 95th percentile over 300 " +
      "single requests), throughput in batches, Gemini spend, and a compute-cost estimate at a configurable price per vCPU-hour, " +
      "per training run and per million recommendations.")));

  root.append(card("Feed influence",
    p("In each session the app shows a feed of 10 dishes chosen by a simple 'old production' rule: popular nearby, a couple of the " +
      "user's past dishes, some random. Real people order partly from the feed and partly by browsing or searching. Feed influence " +
      "is how much more attractive a dish becomes just because the feed showed it. At the current setting a shown dish is about " +
      "2.7× as likely to be picked, and about 15% of orders come from the feed."),
    p("Why it matters: click logs only describe what the old rule chose to show, so a model trained on clicks alone copies its bias. " +
      "The lab learns from orders, whatever path led to them, and uses feed clicks only as a second signal for the DCNv2 ranker.")));

  root.append(card("What is embedded, and when",
    p("Each dish becomes one line of text: dish name, description, cuisine, category, diet flags, tags and restaurant name. " +
      "For example: “Spicy Miso Tantanmen. A bold and fiery broth with minced pork, bok choy and scallions. Ramen & Japanese main " +
      "(spicy). From Komorebi Ramen Gràcia.” Only dishes are embedded. A user's vector is the time-weighted average of the " +
      "vectors of dishes they ordered, so recent orders count more."),
    table(["When", "What happens"], [
      ["Build (done)", "Dish vectors from three backends (TF-IDF, local BGE, Gemini) stored in Postgres with an HNSW index (pgvector)"],
      ["Candidate stage", "pgvector finds the 100 dishes closest to the user's vector among restaurants open and delivering to them. This is what carries a traveller's taste to the other city"],
      ["Two-tower model", "The dish vector is an input to the dish side of the network; the trained network's own vectors are stored and searched the same way"],
      ["Comparison", "Backends compared on quality, time and cost; pgvector exact vs HNSW vs IVFFlat benchmark"]])));

  root.append(card("Honest limits",
    p("The data is simulated. The hidden model decides how users behave, so algorithms whose structure resembles it have an " +
      "advantage. The lab shows how algorithms trade quality, diversity and cost under stated assumptions; it does not predict " +
      "the numbers a real launch would reach.")));
}
