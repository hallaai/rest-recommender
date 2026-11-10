import { api, getMeta, h, fmt, withStates, debounce } from "../app.js";
import { pager } from "./restaurants.js";

export async function render(root, { signal }) {
  const meta = await getMeta();
  const label = Object.fromEntries(meta.cuisines.map((c) => [c.key, c.label]));
  const state = { city: "", cuisine: "", category: "", vegetarian: "", q: "", offset: 0 };
  const list = h("div", { class: "card" });
  const load = () => withStates(list, () => {
    const p = { limit: 50, offset: state.offset };
    for (const k of ["city", "cuisine", "category", "vegetarian", "q"]) if (state[k]) p[k] = state[k];
    return api(`/api/items?${new URLSearchParams(p)}`, { signal });
  }, (d) => {
    list.append(h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Dish", "Category", "Restaurant", "City", "Cuisine", "Price", "Sold"].map((x) => h("th", {}, x)))),
      h("tbody", {}, d.rows.map((i) => h("tr", {},
        h("td", {}, h("div", {}, i.name, " ", i.vegetarian ? h("span", { class: "badge" }, "veg") : null, " ", i.spicy ? h("span", { class: "badge" }, "spicy") : null),
          h("div", { class: "muted small" }, i.description)),
        h("td", {}, i.category), h("td", {}, h("a", { href: `#/restaurants/${i.restaurant_id}` }, i.restaurant)),
        h("td", {}, i.city), h("td", {}, label[i.cuisine]), h("td", { class: "num" }, fmt.eur(i.price)),
        h("td", { class: "num" }, fmt.int(i.orders))))))),
    pager(d.total, state, load));
  });
  const set = (k) => (e) => { state[k] = e.target.value; state.offset = 0; load(); };
  root.append(h("div", { class: "page-head" }, h("h1", {}, "Products"),
    h("p", { class: "lead" }, "Every dish, side, drink and dessert. Mains are what the recommender ranks; add-ons come with them in the basket.")),
  h("div", { class: "filters" },
    h("select", { onchange: set("city"), "aria-label": "City" }, h("option", { value: "" }, "Both cities"), meta.cities.map((c) => h("option", { value: c.code }, c.name))),
    h("select", { onchange: set("cuisine"), "aria-label": "Cuisine" }, h("option", { value: "" }, "All cuisines"), meta.cuisines.map((c) => h("option", { value: c.key }, c.label))),
    h("select", { onchange: set("category"), "aria-label": "Category" }, h("option", { value: "" }, "All categories"), meta.categories.map((c) => h("option", { value: c }, c))),
    h("select", { onchange: set("vegetarian"), "aria-label": "Diet" }, h("option", { value: "" }, "Any diet"), h("option", { value: "true" }, "Vegetarian"), h("option", { value: "false" }, "Not vegetarian")),
    h("input", { type: "search", placeholder: "Search dish or description", oninput: debounce(set("q")) })), list);
  load();
}
