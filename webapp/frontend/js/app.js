// Router, API client and small DOM helpers shared by every view.
import * as overview from "./views/overview.js";
import * as mapView from "./views/map.js";
import * as restaurants from "./views/restaurants.js";
import * as products from "./views/products.js";
import * as users from "./views/users.js";
import * as later from "./views/later.js";
import * as method from "./views/method.js";
import * as compare from "./views/compare.js";
import * as costs from "./views/costs.js";
import * as recommend from "./views/recommend.js";

const routes = {
  overview, map: mapView, restaurants, products, users,
  recommend, compare, costs, method,
};

let current = null;   // AbortController of the view being shown
let meta = null;

export class ApiError extends Error {}

export async function api(path, { signal, method = "GET", body } = {}) {
  const res = await fetch(path, {
    method, signal, credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  // Session expired or missing: the gate wants a fresh login. Go to the login page (it owns auth).
  if (res.status === 401 && path !== "/api/login") { location.replace("/login.html"); throw new ApiError("Login required."); }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try { msg = (await res.json()).detail || msg; } catch { /* keep default */ }
    if (typeof msg !== "string") msg = `Invalid request (${res.status})`;
    throw new ApiError(msg);
  }
  return res.json();
}

export async function getMeta() {
  if (!meta) meta = await api("/api/meta");
  return meta;
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;      // only ever used with static strings
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const fmt = {
  int: (v) => Number(v ?? 0).toLocaleString("en-US"),
  eur: (v) => `€${Number(v ?? 0).toFixed(2)}`,
  pct: (v, d = 1) => `${(Number(v ?? 0) * 100).toFixed(d)}%`,
  date: (v) => (v ? String(v).slice(0, 10) : "–"),
  dt: (v) => (v ? String(v).slice(0, 16).replace("T", " ") : "–"),
};

export function stateBox(kind, text, retry) {
  return h("div", { class: `state ${kind}` }, text, retry ? h("div", {}, h("button", { onclick: retry, style: "margin-top:10px" }, "Retry")) : null);
}

// Runs an async loader with loading / error-with-retry states inside `el`.
export async function withStates(el, loader, render) {
  const run = async () => {
    el.replaceChildren(stateBox("loading", "Loading…"));
    try {
      const data = await loader();
      el.replaceChildren();
      render(data);
    } catch (e) {
      if (e.name === "AbortError") return;
      el.replaceChildren(stateBox("error", e instanceof ApiError ? e.message : "Could not reach the server.", run));
    }
  };
  await run();
}

export function debounce(fn, ms = 300) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function familyColor(family, families) {
  const i = families.indexOf(family);
  return `var(--s${(i % 8) + 1})`;
}

function route() {
  const [name, ...rest] = (location.hash.replace(/^#\/?/, "") || "overview").split("/");
  const mod = routes[name] || overview;
  current?.abort();
  current = new AbortController();
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#/${name}`));
  const view = document.getElementById("view");
  view.replaceChildren();
  window.scrollTo(0, 0);
  mod.render(view, { params: rest.map(decodeURIComponent), signal: current.signal, name });
}

function initTheme() {
  const btn = document.getElementById("theme");
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch { /* storage blocked */ }
  // Default to dark when the viewer has made no explicit choice; the toggle still wins.
  document.documentElement.dataset.theme = saved || "dark";
  btn.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* ignore */ }
    window.dispatchEvent(new Event("themechange"));
  });
}

initTheme();
document.getElementById("logout")?.addEventListener("click", async () => {
  try { await fetch("/api/logout", { method: "POST", credentials: "same-origin" }); } catch { /* ignore */ }
  location.replace("/login.html");
});
window.addEventListener("hashchange", route);
route();
