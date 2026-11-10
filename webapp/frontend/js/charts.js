// Small SVG charts: grouped bars (vertical or horizontal), lines, heatmap.
// Thin marks, rounded data ends, recessive grid, hover tooltips; colours are CSS tokens.
const NS = "http://www.w3.org/2000/svg";

function s(tag, attrs = {}) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

const tip = () => document.getElementById("tip");
export function showTip(evt, html) {
  const t = tip();
  t.replaceChildren();
  for (const line of html) {
    const div = document.createElement("div");
    if (Array.isArray(line)) {
      const [color, label, value] = line;
      if (color) { const d = document.createElement("span"); d.className = "dot"; d.style.background = color; div.append(d); }
      div.append(`${label}: `);
      const b = document.createElement("b"); b.textContent = value; div.append(b);
    } else div.textContent = line;
    t.append(div);
  }
  t.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = `${x}px`; t.style.top = `${y}px`;
}
export const hideTip = () => { tip().hidden = true; };

// Draw at the container's real width so 11px text stays 11px in narrow cards.
function width(el) {
  return Math.max(320, Math.round(el.clientWidth || 640));
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

function legend(series) {
  const el = document.createElement("div");
  el.className = "chart-legend";
  if (series.length < 2) return el;
  for (const sr of series) {
    const sp = document.createElement("span");
    const d = document.createElement("span"); d.className = "dot"; d.style.background = sr.color;
    sp.append(d, sr.name);
    el.append(sp);
  }
  return el;
}

// Rounded-end bar path: rounded at the data end, square at the baseline.
function barPath(x, y, w, hgt, horizontal) {
  const r = Math.min(4, (horizontal ? hgt : w) / 2, horizontal ? w : hgt);
  if (hgt <= 0 || w <= 0) return "";
  if (horizontal) return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt - r}Q${x + w},${y + hgt} ${x + w - r},${y + hgt}H${x}Z`;
  return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`;
}

// opts: { categories, series: [{name, values, color}], horizontal, format, height, labelWidth }
export function bars(el, opts) {
  const { categories, series, horizontal = false, format = (v) => v } = opts;
  const W = width(el), n = categories.length, k = series.length;
  const max = niceMax(Math.max(...series.flatMap((x) => x.values), 0));
  const svg = s("svg", {});
  if (horizontal) {
    const lw = opts.labelWidth || 150, row = k * 12 + 10, H = n * row + 24, plotW = W - lw - 50;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    categories.forEach((c, i) => {
      const y0 = i * row + 4;
      const t = s("text", { x: lw - 8, y: y0 + (k * 12) / 2 + 4, "text-anchor": "end", class: "lbl" }); t.textContent = c; svg.append(t);
      series.forEach((sr, j) => {
        const v = sr.values[i] || 0, w = (v / max) * plotW, y = y0 + j * 12;
        svg.append(s("path", { d: barPath(lw, y, w, 10, true), fill: sr.color }));
        if (k === 1) { const vt = s("text", { x: lw + w + 6, y: y + 9, class: "val" }); vt.textContent = format(v); svg.append(vt); }
      });
      const hit = s("rect", { x: 0, y: y0 - 3, width: W, height: row, class: "hit" });
      hit.addEventListener("mousemove", (e) => showTip(e, [c, ...series.map((sr) => [sr.color, sr.name, format(sr.values[i] || 0)])]));
      hit.addEventListener("mouseleave", hideTip);
      svg.append(hit);
    });
    svg.append(s("line", { x1: lw, x2: lw, y1: 0, y2: n * row, class: "baseline" }));
  } else {
    const H = opts.height || 220, pb = 26, pl = 44, plotH = H - pb - 8, plotW = W - pl - 4;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    for (let g = 0; g <= 4; g++) {
      const y = 8 + plotH - (g / 4) * plotH;
      svg.append(s("line", { x1: pl, x2: W, y1: y, y2: y, class: g ? "gridline" : "baseline" }));
      const t = s("text", { x: pl - 6, y: y + 4, "text-anchor": "end", class: "lbl" }); t.textContent = format((max * g) / 4); svg.append(t);
    }
    const slot = plotW / n, gap = 2, bw = Math.max(2, Math.min(28, (slot * 0.75 - gap * (k - 1)) / k));
    const every = Math.ceil(n / 16);
    categories.forEach((c, i) => {
      const x0 = pl + i * slot + (slot - (bw * k + gap * (k - 1))) / 2;
      series.forEach((sr, j) => {
        const v = sr.values[i] || 0, hh = (v / max) * plotH;
        svg.append(s("path", { d: barPath(x0 + j * (bw + gap), 8 + plotH - hh, bw, hh, false), fill: sr.color }));
      });
      if (i % every === 0) { const t = s("text", { x: pl + i * slot + slot / 2, y: H - 8, "text-anchor": "middle", class: "lbl" }); t.textContent = c; svg.append(t); }
      const hit = s("rect", { x: pl + i * slot, y: 0, width: slot, height: H, class: "hit" });
      hit.addEventListener("mousemove", (e) => showTip(e, [c, ...series.map((sr) => [sr.color, sr.name, format(sr.values[i] || 0)])]));
      hit.addEventListener("mouseleave", hideTip);
      svg.append(hit);
    });
  }
  const wrap = document.createElement("div"); wrap.className = "chart";
  wrap.append(legend(series), svg);
  el.replaceChildren(wrap);
}

// opts: { x: [labels], series: [{name, values, color}], format, height, yMin }
export function lines(el, opts) {
  const { x, series, format = (v) => v } = opts;
  const W = width(el), H = opts.height || 240, pb = 26, pl = 48, plotH = H - pb - 8, plotW = W - pl - 10;
  const vals = series.flatMap((sr) => sr.values.filter((v) => v !== null));
  const max = niceMax(Math.max(...vals, 0)), min = opts.yMin ?? 0;
  const X = (i) => pl + (x.length === 1 ? plotW / 2 : (i / (x.length - 1)) * plotW);
  const Y = (v) => 8 + plotH - ((v - min) / (max - min)) * plotH;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
  for (let g = 0; g <= 4; g++) {
    const v = min + ((max - min) * g) / 4, y = Y(v);
    svg.append(s("line", { x1: pl, x2: W, y1: y, y2: y, class: g ? "gridline" : "baseline" }));
    const t = s("text", { x: pl - 6, y: y + 4, "text-anchor": "end", class: "lbl" }); t.textContent = format(v); svg.append(t);
  }
  const every = Math.ceil(x.length / 12);
  x.forEach((c, i) => { if (i % every === 0) { const t = s("text", { x: X(i), y: H - 8, "text-anchor": "middle", class: "lbl" }); t.textContent = c; svg.append(t); } });
  for (const sr of series) {
    const d = sr.values.map((v, i) => (v === null ? null : `${X(i)},${Y(v)}`)).filter(Boolean).join(" L");
    svg.append(s("path", { d: `M${d}`, fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round" }));
  }
  const cross = s("line", { y1: 8, y2: 8 + plotH, class: "gridline", visibility: "hidden" });
  svg.append(cross);
  const hit = s("rect", { x: pl, y: 0, width: plotW, height: H, class: "hit" });
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(x.length - 1, Math.round(((px - pl) / plotW) * (x.length - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    showTip(e, [x[i], ...series.map((sr) => [sr.color, sr.name, sr.values[i] === null ? "–" : format(sr.values[i])])]);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); });
  svg.append(hit);
  const wrap = document.createElement("div"); wrap.className = "chart";
  wrap.append(legend(series), svg);
  el.replaceChildren(wrap);
}

// opts: { rows, cols, values (rows x cols, 0..1 or null), format, cellH }
export function heatmap(el, opts) {
  const { rows, cols, values, format = (v) => v.toFixed(2) } = opts;
  const lw = opts.labelWidth || 150, W = width(el), cw = (W - lw) / cols.length, ch = opts.cellH || 26, top = 70;
  const H = top + rows.length * ch + 4;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
  const steps = ["--seq-1", "--seq-3", "--seq-5", "--seq-7", "--seq-9"];
  cols.forEach((c, j) => {
    const t = s("text", { x: lw + j * cw + cw / 2, y: top - 8, "text-anchor": "start", class: "lbl", transform: `rotate(-35 ${lw + j * cw + cw / 2} ${top - 8})` });
    t.textContent = c; svg.append(t);
  });
  rows.forEach((r, i) => {
    const t = s("text", { x: lw - 8, y: top + i * ch + ch / 2 + 4, "text-anchor": "end", class: "lbl" }); t.textContent = r; svg.append(t);
    cols.forEach((c, j) => {
      const v = values[i][j];
      const fill = v === null ? "var(--surface-2)" : `var(${steps[Math.min(4, Math.floor(v * 5))]})`;
      const rect = s("rect", { x: lw + j * cw + 1, y: top + i * ch + 1, width: cw - 2, height: ch - 2, rx: 3, fill });
      rect.addEventListener("mousemove", (e) => showTip(e, [`${r} × ${c}`, ["", "value", v === null ? "–" : format(v)]]));
      rect.addEventListener("mouseleave", hideTip);
      svg.append(rect);
    });
  });
  const wrap = document.createElement("div"); wrap.className = "chart"; wrap.append(svg);
  el.replaceChildren(wrap);
}
