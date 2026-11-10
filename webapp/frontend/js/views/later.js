import { h } from "../app.js";

const TEXT = {
  recommend: "Pick a user (or a point on the map for a new user), a time and the weather, and see the top 5 from every algorithm side by side, with reasons and latency.",
};

export function render(root, { name }) {
  root.append(h("div", { class: "page-head" }, h("h1", {}, name[0].toUpperCase() + name.slice(1)),
      h("span", { class: "badge", style: "margin-left:8px;vertical-align:middle" }, "phase 3")),
    h("div", { class: "card" },
      h("p", { class: "lead" }, TEXT[name] || ""),
      h("p", { class: "note" },
        "Empty on purpose. This page needs the algorithms loaded into the running server to produce a live top-5 for " +
        "one chosen user — that is phase 3/4 work. What already works today is the Compare page: it ranks the same " +
        "algorithms offline over the whole October test month, with quality, overlap and cost. Recommend will add the " +
        "per-user, live view on top of that."),
      h("p", {}, h("a", { href: "#/compare" }, "→ See the algorithm comparison (Compare)"), " · ",
        h("a", { href: "#/method" }, "How the algorithms work (Method)"))));
}
