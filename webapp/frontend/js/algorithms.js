// Single source of truth for how each algorithm is described in the UI.
// stage: "retrieval" (proposes candidates) or "ranking" (orders candidates) or "both".
export const ALGORITHMS = {
  random: {
    name: "Random", stage: "baseline",
    what: "Shuffles the available dishes. The floor: any real algorithm must beat it.",
    signal: "nothing", news: "no",
  },
  popularity: {
    name: "Popularity", stage: "retrieval",
    what: "Ranks dishes by how often they were ordered overall. The simplest baseline.",
    signal: "global order counts", news: "yes — recommends popular dishes you have not tried",
  },
  geo_time_pop: {
    name: "Geo + time popularity", stage: "retrieval / cold start",
    what: "Popularity within the request's city and time of day (breakfast, lunch, dinner…), " +
          "blended with city-wide and global popularity so a thin slot still ranks sensibly. " +
          "This is the standard answer for a brand-new user with no history.",
    signal: "location, hour, order counts", news: "yes",
  },
  buy_again: {
    name: "Buy-again", stage: "retrieval",
    what: "Scores a dish by how often and how recently the user ordered that exact dish " +
          "(30-day half-life). Only ever suggests dishes the user has already had.",
    signal: "the user's own order history", news: "NO — re-ordering only, never discovery",
  },
  item_item: {
    name: "Item-to-item", stage: "retrieval",
    what: "“People who ordered what you order also order X.” Cosine similarity between dishes " +
          "based on who ordered them; a user's score for a candidate is its summed similarity " +
          "to the dishes they have ordered.",
    signal: "co-occurrence in order histories", news: "yes — related dishes you have not tried",
  },
  ease: {
    name: "EASE", stage: "retrieval",
    what: "A closed-form item-to-item collaborative filter (one matrix inversion). Learns a " +
          "dish-to-dish weight matrix from the whole user×dish order matrix; strong and cheap to serve.",
    signal: "the full order matrix (collaborative filtering)", news: "yes",
  },
  als: {
    name: "ALS matrix factorisation", stage: "retrieval",
    what: "Implicit-feedback matrix factorisation: learns a short vector for every user and every " +
          "dish so their dot product predicts ordering. The classic collaborative-filtering workhorse.",
    signal: "the full order matrix (collaborative filtering)", news: "yes",
  },
  pipeline: {
    name: "Fused pipeline (production path)", stage: "retrieval + ranking",
    what: "The real two-stage system: all six retrieval sources propose candidates, they are pooled " +
          "to ~100 dishes, ~35 features per (user, dish) are built — distance, delivery fee, time fit, " +
          "price fit, history, each retrieval score — and a LightGBM listwise ranker picks the final 5 " +
          "(≤2 per restaurant). This is what beats every single algorithm on the Compare page.",
    signal: "every source above + distance, time, price, history", news: "yes — the best of all of them",
  },
  // phase 3 (not yet built) — shown greyed in Method
  content_ann: {
    name: "Text-embedding ANN", stage: "retrieval (phase 3)",
    what: "Embeds each dish's text (name, description, cuisine…) into a vector; the user's vector is " +
          "the time-weighted average of the dishes they ordered. pgvector returns the nearest dishes. " +
          "Carries taste to new dishes and across cities (a traveller's sushi habit → sushi in the other city).",
    signal: "dish descriptions + the user's past dishes", news: "yes — unseen dishes by meaning",
  },
  two_tower: {
    name: "Two-tower network", stage: "retrieval (phase 3)",
    what: "A neural net with a user tower and a dish tower trained so matching pairs are close. " +
          "Combines many features into learned vectors, searched with pgvector.",
    signal: "many user and dish features", news: "yes",
  },
  ranker: {
    name: "LightGBM / DCNv2 ranker", stage: "ranking (phase 3)",
    what: "Takes the ~100 candidates from the retrieval sources and scores each with ~40 features per " +
          "(user, dish) pair — distance, delivery fee, time fit, price fit, history counts, popularity, " +
          "each retrieval score — to pick the final top 5. This is where distance, price and time enter.",
    signal: "~40 features per (user, dish) pair", news: "orders the candidates; adds no new ones",
  },
};

export const algoName = (k) => (ALGORITHMS[k]?.name) || k;
