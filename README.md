# Food Recs Lab

A lab for food-delivery recommendations: synthetic but realistic data for Barcelona and
Madrid, several recommendation algorithms, and a web app to explore the data and compare
the algorithms' quality, how differently they recommend, and their cost. The task is to show
each user the five dishes that fit on one phone screen. The audience is managers and ML/DS
staff weighing the options.

## Launch

One command, run from `webapp/`:

```
cd webapp
pip install -e .[test]        # once
python -m recsys              # build whatever is missing or out of date, then serve http://127.0.0.1:8000
```

Each step is skipped when its output is already up to date, so the second launch starts in seconds:

| Step | Skipped when | First run |
|---|---|---|
| 1. World (restaurants, menus, users, 6 months of sessions and orders) to `data/` | `data/meta.json` exists. A world built with template menus is regenerated once Gemini is reachable | 1-6 min; about $1 of Gemini for ~350 menus (cached) |
| 2. Dish embeddings (TF-IDF and Gemini; BGE on request) to `data/embeddings/` | built from the current dishes | seconds; Gemini about $0.02. BGE adds 15-30 min on CPU |
| 3. Postgres load (schema `recsys`, PostGIS, pgvector HNSW) | the database's stored fingerprint matches the files | about 1 min |
| 4. Web server | – | – |

Options: `--rebuild` (regenerate everything), `--no-gemini` (never call Gemini),
`--no-serve` (build only), `--port 8001`, `--backends tfidf bge gemini` (add the local BGE model).
If the database is unreachable, the app still starts, shows the database as unavailable, and
reconnects on its own when it comes back.

To evaluate the recommendation algorithms (powers the Compare and Costs pages):

```
python -m recsys.eval.runner              # all baselines, full October test set -> artifacts/leaderboard.json
python -m recsys.eval.runner --sample 3000   # faster: a user-stratified sample
python -m recsys.eval.rank_runner            # train the 4 rankers + the two-stage pipeline -> artifacts/ranking.json

# phase-5 experiments (each writes one artifact the Compare page reads; all optional, all heavy):
python -m recsys.eval.ann_bench              # pgvector exact vs HNSW vs IVFFlat recall/latency -> artifacts/ann.json
python -m recsys.eval.scaling                # cost/latency projection to 10x/100x catalogue -> artifacts/scaling.json
python -m recsys.eval.bandit                 # Thompson cold-start bandit vs static policy -> artifacts/bandit.json
python -m recsys.eval.gemini_rerank --requests 60   # LLM rerank of the top candidates (paid) -> artifacts/gemini_rerank.json
```

The Recommend page shows **the fused pipeline's five dishes as the headline recommendation** (a
full-width hero at the top), with every individual method below as "how each would choose on its
own" — so you can see what the fusion is built from. The pipeline warms in the background (tens of
seconds after first load); until then the hero shows a warming notice. Each card carries two
buy-probability badges: **avg** (the method's Demand@5 over the October test set, from the offline
eval) and **buy-prob** (the live true buy-probability of the five dishes it just showed this user,
from the simulation's hidden truth — a demo-only lens, since production cannot see those
probabilities; absent when `data/truth/` is not present).

Settings live in `webapp/.env` (template: `webapp/.env.example`): `DATABASE_URL`, `GEMINI_API_KEY`,
`APP_USERNAME` (default `admin`) and `APP_PASSWORD` (empty = no login, local dev only), and
Gemini spend caps. **With a password set, nothing is served until login:** the gate withholds
every page, script, stylesheet and API response from unauthenticated callers and returns only
the login page, so a scraper sees no data and no application code. Login is throttled per client
address; "Sign out" clears the session. Tests need neither the database nor
keys: `python -m pytest -q`.

## Status

| Phase | What | State |
|---|---|---|
| 0 | New `recsys` package, Postgres schema `recsys` (pgvector + PostGIS), git | done |
| 1 | Synthetic world, embeddings, database load, Explore UI (overview, map, restaurants, products, users) | done |
| 2 | Evaluation harness, metrics with CIs, cost measurement, baselines; Compare + Costs pages | done |
| 3 | Content/ANN retriever, two-tower (profile features for cold start), LightGBM ranker (3 losses), DCNv2, fused two-stage pipeline | done |
| 4 | Recommend page (live top-5 per algorithm, incl. the live fused pipeline row) | done |
| 5 | Cold-start bandit, Gemini reranker, pgvector ANN benchmark, scaling projection | done |

## The synthetic world

| | Size | Notes |
|---|---|---|
| Restaurants | 400 (180 BCN, 220 MAD) | Real district centroids. Price tier follows district income. Spanish opening hours, some closed between lunch and dinner. 7 fictional chains in both cities with the same menu |
| Dishes | about 8.7k (about 5k mains) | Menus written by Gemini per restaurant: fictional names, descriptions, prices, veg/vegan/spicy flags |
| Users | 6,000 | Visible profile: home, work, signup date, platform, channel, age. Hidden taste: `recsys/synth/truth.py` |
| Sessions | about 150k, about 62% order | Each session shows a 10-dish feed; see *Feed influence* |
| Orders | about 95k, about €16 and 2 items | About 28% repeat a dish, 45% repeat a restaurant. About 10% of users travel to the other city |
| Time | 2026-05-01 to 2026-10-31 | Train May to Aug, validation Sep, test Oct. August dips, October grows with new signups |

Every simulated user has a hidden taste: favourite cuisines, a personal liking for individual
dishes, sensitivity to price and distance, diet, spice tolerance, reorder habit and meal times.
In each session the user weighs those against the dish, the restaurant, distance, delivery
fee, time of day and weather. The evaluator uses the same hidden model to score any
recommendation by the true chance the user would order it. **Recommendation models never
read it.** Results describe this simulator's assumptions, not a production forecast.

### Feed influence

The app shows a feed of 10 dishes in each session, chosen by a simple "old production"
policy: popular dishes nearby, a couple of the user's past dishes, and some random ones.
Real users order partly from the feed and partly by browsing or searching. *Feed
influence* (`EXPOSURE_BOOST` in `truth.py`) is how much more attractive a dish becomes simply
because the feed showed it. At the current value (1.0, about 2.7 times as likely) **about 15% of
orders come from the feed** and 85% from browsing.

Why it matters: impression and click logs only describe what the old policy chose to show.
A model trained on clicks alone learns that policy's bias. The lab therefore trains on orders,
whatever path led to them, and uses impressions only for the click part of the multi-task
ranker (DCNv2, phase 3).

### What is embedded, and when

Each dish becomes one line of text (`recsys/embed.py: item_texts`):

```
Spicy Miso Tantanmen. A bold and fiery broth with minced pork, bok choy and scallions.
Ramen & Japanese main (spicy). ramen, spicy. From Komorebi Ramen Gràcia.
```

That is dish name, description, cuisine, category, diet flags, tags and restaurant name. Only
dishes are embedded; users are not. A user's vector is computed from the dishes they ordered.

| Phase | Use |
|---|---|
| 1 (done) | Vectors computed with three backends and stored in Postgres tables `vec_tfidf`, `vec_bge`, `vec_gemini`, each with an HNSW index |
| 3 | **Content retrieval:** user vector = time-weighted average of the vectors of dishes they ordered (recent orders count more). pgvector returns the 100 nearest dishes among restaurants that are open and deliver to the user. This is what carries a traveller's taste to the other city and handles new dishes |
| 3 | **Two-tower model:** the dish vector is an input to the item tower. The trained tower's own 64-d vectors go to `vec_two_tower` |
| 3-5 | Compared on quality, time and cost, plus a pgvector benchmark (exact vs HNSW vs IVFFlat) |

| Backend | Dim | Time for 8.7k dishes | Cost |
|---|---|---|---|
| TF-IDF + SVD | 128 | about 6 s | free |
| BGE base (local CPU, opt-in) | 768 | 15-30 min (6 min with short template texts) | free (compute only) |
| Gemini embedding-001 | 768 | see `data/embeddings/gemini.json` | about $0.02 |

## How features are used — the two-stage design

A recommendation is not made by one model, and text-embedding similarity is only one signal.
If it went *only* by dish-description similarity (ANN over embeddings) it would ignore distance,
price, time, the user's order history, popularity, and whether the restaurant is even open.

**Stage 1 — Candidates (~100 dishes).** Several retrieval sources each propose dishes; the
embedding/ANN path is one of them, used mainly to carry taste to unseen dishes and across cities:

| Source | Signal it uses | Suggests dishes the user never ordered? |
|---|---|---|
| Geo + time popularity | location, hour, open-now | yes (cold-start answer) |
| Buy-again | the user's own history (recency × frequency) | **no — re-ordering only** |
| Item-to-item | co-occurrence in order histories | yes |
| EASE / ALS | the full user×dish order matrix (CF) | yes |
| Text-embedding ANN *(phase 3)* | dish descriptions + the user's past dishes | yes (by meaning) |
| Two-tower *(phase 3)* | many learned user+dish features | yes |

**Stage 2 — Ranking *(phase 3)*.** The pooled ~100 candidates are scored by LightGBM (then DCNv2)
with ~40 features per (user, dish) pair — distance, delivery fee, time fit, price fit, history
counts (7/30/90-day), zone popularity, each retrieval score, rating, is-reorder, diet match.
**ANN never decides the final list alone:** a dish with a perfect description match but 8 km away
at a closed restaurant is filtered (availability) or ranked down (distance feature).

**Buy-again vs the rest:** buy-again can only recommend dishes the user already ordered; every
other source recommends *new* dishes (discovery). This is why buy-again dominates for heavy users
and collapses to near-zero for cold users — visible in the Compare page's segment breakdown.

Phase 2 (below) measures the retrieval sources on their own. The ANN retriever, two-tower and
the ranker are phase 3.

## Phase 2 results (22,141 October requests)

Retrieval baselines, ranked by true demand captured@5 (the chance the user orders one of the 5
shown, from the hidden model). These are **retrieval sources measured on their own**; the
two-stage pipeline (phase 3) pools candidates from several of them and re-ranks.

| Algorithm | Demand@5 | HitRate@5 | % of best | Fit | p95 latency |
|---|---|---|---|---|---|
| Buy-again | 0.274 | 0.301 | 77% | 0 s | 0.2 ms |
| ALS | 0.200 | 0.219 | 56% | 119 s | 1.9 ms |
| Item-to-item | 0.153 | 0.172 | 43% | 1 s | 1.4 ms |
| EASE | 0.097 | 0.111 | 28% | 7 s | 59 ms |
| Geo + time popularity | 0.065 | 0.068 | 18% | 0 s | 0.4 ms |
| Popularity | 0.063 | 0.066 | 18% | 0 s | 0.3 ms |
| Random | 0.010 | 0.009 | 3% | 0 s | 0.3 ms |

Key findings, visible on the **Compare** page's segment charts:
- **Buy-again leads overall but only for users with history.** For cold users (no orders) it is
  near-zero, like everything else — the cold-start gap.
- **Every algorithm collapses in the other city** (travellers): little local history for CF to use.
  This is the gap the phase-3 embedding retriever is meant to close.
- **Cost trade-offs differ sharply:** ALS is expensive to fit (119 s) but cheap to serve; EASE is
  the opposite (slow per request, 786 MB). The Costs page shows this.

## Phase 3 results: the two-stage pipeline (3,000 October requests)

Candidates pooled from six retrieval sources, re-ranked with ~40 features per (user, dish) pair.

| Ranker (loss) | Demand@5 | NDCG@5 | % of best | Cold-start demand | Train |
|---|---|---|---|---|---|
| LightGBM pairwise (LambdaRank) | 0.402 | 0.258 | 73% | 0.075 | 69 s |
| LightGBM listwise (rank_xendcg) | 0.397 | 0.257 | 73% | 0.071 | **29 s** |
| DCNv2 (neural) | 0.396 | 0.234 | 72% | 0.063 | 100 s |
| LightGBM pointwise (logloss) | 0.339 | 0.195 | 62% | 0.083 | 60 s |

**Business findings (do you even need the complex pipeline?):**
- **The fused pipeline (0.40) beats the best single retriever (buy-again 0.27) by ~48%.** Pooling +
  ranking genuinely wins. The ranker's top features mix hard and soft signals: `distance_km`,
  `retr_content` (embedding score), `retr_geo_pop`, `retr_als`, price and recency — side by side.
- **DCNv2 does not beat LightGBM** (0.396 vs 0.402) and costs 3.4x more to train. On tabular
  ranking features, gradient-boosted trees win — so the neural ranker is not worth it here.
- **Listwise (rank_xendcg) is the one to ship:** tied with pairwise on quality, trains in half the time.
- **But the baseline is strong:** buy-again alone reaches 77% of best for the ~75% of users who have
  history. The pipeline earns its cost mainly on the **25% cold-start users** (1,505 of 6,000) and on
  discovery/travellers, where buy-again scores ~0. Whether that segment justifies the complexity is
  the real decision — the lab quantifies both sides.

Of the 6,000 users, before the test month: **1,505 (25%) have no history**, 801 have 1-3 orders,
1,726 have 4-15, 1,968 have 16+. The data holds 95,681 orders and 1.5M feed impressions.

## How algorithms are evaluated (phase 2)

**Unit of evaluation.** Each October order is one request: user, location, time and weather at
order time. The candidates are all main dishes from restaurants that are open and deliver
there, about 1,200. The algorithm ranks them, and the dish the user actually ordered is the
target. Every algorithm sees only orders from before the request's day. Models are fitted on
May-Sep, with hyperparameters chosen on September while fitted on May-Aug.

**Quality metrics (top 5 = one phone screen)**

| Metric | Meaning |
|---|---|
| HitRate@5 | Share of requests where the ordered dish is in the top 5 |
| NDCG@5 | Like HitRate, but a hit at position 1 counts more than at position 5 (1 / log2(position + 1)) |
| MRR | Average of 1 / position of the ordered dish (how high it appears) |
| Recall@100 | Candidate stage only: is the ordered dish among the 100 passed to the ranker |
| Restaurant hit@5 | The top 5 contains a dish from the restaurant the user ordered from |
| **True demand captured@5** | From the hidden model: probability that the user orders one of the 5 recommended dishes. Uses every dish's true probability, not just the one observed order, so it is far less noisy |
| % of best possible | Captured demand ÷ that of the best 5 dishes under the hidden model |
| Expected revenue per 1,000 sessions | Captured demand × price, in EUR |

**Behaviour metrics.** These cover how algorithms differ, not only how well they score:
- catalogue and restaurant coverage
- share of reorders vs new dishes
- distinct restaurants and cuisines in the top 5
- popularity bias (average popularity percentile of recommended dishes)
- exposure inequality across restaurants (Gini coefficient)
- **overlap between algorithms**: average Jaccard similarity of their top-5 lists for the same request
- per-user win and loss rate between pairs of algorithms

**Segments.** Everything is also reported by user history at request time (none, 1-3, 4-15, 16+
orders), for travellers (ordering in the other city), by city and by meal time.

**Confidence intervals.**
- Every metric gets a 95% **bootstrap percentile interval**: 1,000 resamples drawn at the user level (a user and all their requests together), because one user's requests are not independent.
- Differences between two algorithms use a **paired bootstrap** on the same resamples. A gap is called real only when its interval excludes zero.
- Small segments (about 1,000 cold-start and about 250 traveller requests) get wide intervals, and the app shows them.

**Cost metrics:**
- training wall time and CPU time, and peak memory
- model size
- latency per request (p50 / p95 over 300 single requests) and batch throughput
- Gemini spend
- a compute cost estimate in USD at a configurable vCPU-hour price (`VCPU_HOUR_USD`), per training run and per million recommendations

## Layout

```
webapp/
  recsys/build.py  one-command build and serve (python -m recsys)
  recsys/synth/    world generator: geo, restaurants, menus (Gemini), users, truth, simulation
  recsys/store/    Postgres schema, loader (COPY, fingerprint), read-only repository
  recsys/api/      FastAPI: /api/* (docs at /api/docs) plus static UI
  recsys/embed.py  dish text embeddings (tfidf, bge, gemini)
  recsys/data.py   offline dataset for training and evaluation (point-in-time history)
  frontend/        no-build JS UI (Leaflet map, SVG charts)
  tests/
```

Impressions (1.5M rows) stay in parquet and are not loaded into Postgres, because only offline
training uses them.
