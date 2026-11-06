// Shared helpers for the results report (report.js) and the per-run pages (run.js); both read window.R from data.js.
"use strict";
// Prose layer (text.js, build step `text`) over the numbers (data.js, build step `data`): report text, run names, comparison labels.
Object.assign(R, { meta: T.meta, glossary: T.glossary, sections: T.sections, strategy: T.strategy, methods: T.methods });
for (const [r, [scene, condition]] of Object.entries(T.names)) Object.assign(R.runs[r], { scene, condition });
for (const [cid, c] of Object.entries(T.comparisons)) Object.assign(R.comparisons[cid], c);
for (const items of Object.values(R.examples)) for (const e of items) e.name = T.example_names[e.text];
const RUNS = R.runs, M = R.metrics, $ = id => document.getElementById(id), NS = "http://www.w3.org/2000/svg";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const name = r => `${RUNS[r].scene} · ${RUNS[r].condition}`;
const runHref = r => `run.html?run=${encodeURIComponent(r)}`;
const runLink = (r, text) => `<a href="${runHref(r)}">${text ?? name(r)}</a>`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (v, d = 3) => v == null ? "—" : Math.abs(v) >= 1000 || (Math.abs(v) < 1e-3 && v !== 0) ? (+v).toExponential(1) : (+v).toPrecision(d).replace(/\.?0+$/, m => m.includes(".") ? "" : m);
const pct = v => v == null ? "—" : v < 0.001 ? "exact" : (100 * v).toFixed(v < 0.1 ? 1 : 0) + "%";
const xr = v => v == null ? "—" : v === 0 ? "0 (dropped)" : Math.abs(v - 1) < 1e-3 ? "×1 exact" : "×" + fmt(v, 3);
const mmf = v => v == null ? "—" : v < 0.005 ? "≈0" : fmt(v, 3);
const PALETTE = ["--base", "--abl", "--third", "--fourth", "--s5", "--s6"];
const KEYCOLOR = { "r:mass": "--base", "r:inertia": "--fourth", "r:damping": "--abl", "r:frictionloss": "--third", "r:stiffness": "--s5",
  "r:mass:block_red": "--abl", "r:mass:block_blue": "--base", "r:massratio": "--fourth" };
const label = k => (M[k] || [k.replace("r:mass:block_", "").replace("r:", "") + " mass"])[0];
const kind = k => (M[k] || [0, "ratio"])[1];
const fmtKey = (k, v) => kind(k) === "ratio" ? xr(v) : k.endsWith("_mm") ? mmf(v) : k.endsWith("_rel") || k === "sphere_err" ? pct(v) : fmt(v, 2);
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const html = s => { const t = document.createElement("template"); t.innerHTML = s.trim(); return t.content.firstChild; };
const title = (node, text) => { el("title", {}, node).textContent = text; return node; };
const div = (parent, cls) => { const d = document.createElement("div"); if (cls) d.className = cls; parent.appendChild(d); return d; };
const svgLink = (parent, r) => el("a", { href: runHref(r) }, parent);  // SVG labels that open a run's page
