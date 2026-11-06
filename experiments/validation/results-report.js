// Generic renderer for build-results-report.py: every section, chart, table and 3D overlay comes from window.R (data.js),
// whose sections are declared by the report spec. Inline SVG; no network.
// Helpers (fmt, name, runLink, el, html, ...) come from common.js.

// ---------- generic charts ----------
function ticks(lo, hi, log) {
  if (log) { const out = [], ms = Math.log10(hi / lo) > 3 ? [1] : [1, 2, 5]; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of ms) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); } return out; }
  const span = hi - lo, step0 = span / 5, mag = 10 ** Math.floor(Math.log10(step0)), step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(10)); return out;
}
function lineChart(host, o) {  // series [{name, color, dash, faded, pts:[[x,y]], spread?:[[x,lo,hi]]}]; on log charts y<=floor sits on the floor:
  // hollow = parameter left out (ratio plots), filled = exact (error plots, floor labeled "exact")
  const W = 420, H = o.height || 230, L = 50, Rm = 10, T = o.title ? 24 : 10, Bm = 34, floor = o.floor || 0;
  const pts = o.series.flatMap(s => s.pts.filter(p => p[1] != null));
  if (!pts.length) return;
  const spread = o.series.flatMap(s => (s.spread || []).flatMap(q => [q[1], q[2]]));
  const pos = pts.map(p => p[1]).concat(spread).filter(v => !o.log || v > floor).concat(o.truth != null ? [o.truth] : [], o.band ? o.band.slice(0, 2) : []);
  let y0 = Math.min(...pos), y1 = Math.max(...pos);
  if (o.log) { y0 = Math.max(Math.min(y0, y1 / 4) / 1.6, floor || 1e-4); y1 *= 1.6; } else { const pad = (y1 - y0 || Math.abs(y1) || 1) * 0.12; y0 = Math.max(0, y0 - pad); y1 += pad; }
  const x0 = 0, x1 = o.xmax || Math.max(240, ...pts.map(p => p[0]));
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - Rm);
  const sy = y => o.log ? T + (Math.log(y1) - Math.log(Math.max(y, y0))) / (Math.log(y1) - Math.log(y0)) * (H - T - Bm) : T + (y1 - y) / (y1 - y0) * (H - T - Bm);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": o.title || "" });
  if (o.title) el("text", { x: 4, y: 15, class: "ttl" }, svg).textContent = o.title;
  for (const v of ticks(y0, y1, o.log)) { el("line", { x1: L, x2: W - Rm, y1: sy(v), y2: sy(v), class: "gr" }, svg); el("text", { x: L - 6, y: sy(v) + 4, "text-anchor": "end" }, svg).textContent = (o.ratio ? "×" : "") + fmt(v, 2); }
  for (const v of ticks(x0, x1, false)) el("text", { x: sx(v), y: H - Bm + 15, "text-anchor": "middle" }, svg).textContent = fmt(v, 3);
  el("line", { x1: L, x2: W - Rm, y1: H - Bm, y2: H - Bm, class: "ax" }, svg);
  el("text", { x: (L + W) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = "minutes into run";
  if (o.band) title(el("rect", { x: L, width: W - L - Rm, y: sy(o.band[1]), height: Math.max(1, sy(o.band[0]) - sy(o.band[1])), fill: css("--base"), opacity: 0.13 }, svg), o.band[2]);
  if (o.truth != null) title(el("line", { x1: L, x2: W - Rm, y1: sy(o.truth), y2: sy(o.truth), class: "truth" }, svg), "true value");
  if (o.at120) el("line", { x1: sx(120), x2: sx(120), y1: T, y2: H - Bm, class: "gr", "stroke-dasharray": "2 3" }, svg);
  if (o.log && floor && !o.ratio && pts.some(p => p[1] <= floor)) el("text", { x: L - 6, y: sy(y0) + 4, "text-anchor": "end", style: "fill:var(--good)" }, svg).textContent = "exact";
  for (const s of o.series) {
    const p = s.pts.filter(q => q[1] != null), color = css(s.color);
    if (s.spread && s.spread.length > 1)  // shaded range across the condition's runs
      el("path", { d: s.spread.map((q, i) => (i ? "L" : "M") + sx(q[0]).toFixed(1) + "," + sy(q[2]).toFixed(1)).join("") +
        s.spread.slice().reverse().map(q => "L" + sx(q[0]).toFixed(1) + "," + sy(q[1]).toFixed(1)).join("") + "Z", fill: color, opacity: 0.16, stroke: "none" }, svg);
    if (p.length > 1) el("path", { d: p.map((q, i) => (i ? "L" : "M") + sx(q[0]).toFixed(1) + "," + sy(q[1]).toFixed(1)).join(""), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": s.dash || "", opacity: s.faded ? 0.45 : 1 }, svg);
    for (const q of p) {
      const dropped = o.log && o.ratio && q[1] <= floor;  // exact zeros on error plots are filled, on the "exact" floor
      title(el("circle", { cx: sx(q[0]), cy: sy(q[1]), r: s.spread ? 2.5 : 4, fill: dropped ? css("--card") : color, stroke: dropped ? color : css("--card"), "stroke-width": dropped ? 2 : 1.5, opacity: s.faded ? 0.55 : 1 }, svg),
        `${s.name}: ${s.spread ? "mean at" : "submission at"} ${q[0].toFixed(0)} min = ${o.fmtv(q[1])}`);
    }
  }
  host.appendChild(svg);
}
function legend(host, items) {
  host.innerHTML = items.map(([lab, color, k]) => `<span><i class="sw ${k || ""}" style="background:${color ? css(color) : ""}"></i>${lab}</span>`).join("");
}
function stackedBars(host, rows, cats, o = {}) {  // rows: [{label, values:{cat:n}}], 100% stacked
  const W = 820, rowH = 18, Lw = 340, T = 4, H = T + rows.length * (rowH + 5) + 4;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
  rows.forEach((r, i) => {
    const y = T + i * (rowH + 5), tot = cats.reduce((a, c) => a + (r.values[c[0]] || 0), 0);
    el("text", { x: Lw - 8, y: y + 13, "text-anchor": "end", style: r.bold ? "font-weight:600;fill:var(--fg)" : "" }, r.run ? svgLink(svg, r.run) : svg).textContent = r.label;
    let x = Lw;
    for (const [c, lab, color] of cats) {
      const w = (r.values[c] || 0) / tot * (W - Lw - 4);
      if (w <= 0) continue;
      title(el("rect", { x: x + 1, y, width: Math.max(0, w - 2), height: rowH, fill: css(color), rx: 2 }, svg), `${r.label} · ${lab}: ${o.unit === "h" ? (r.values[c] / 3600).toFixed(2) + " h" : r.values[c]} (${(100 * r.values[c] / tot).toFixed(0)}%)`);
      if (w > 30) el("text", { x: x + w / 2, y: y + 13, "text-anchor": "middle", style: "fill:#fff;font-size:10.5px" }, svg).textContent = Math.round(100 * r.values[c] / tot) + "%";
      x += w;
    }
  });
  host.appendChild(svg);
}
function hbars(host, b) {  // {title, rows:[{label, v, color}], unit, max}
  const W = 620, rowH = 14, Lw = 330, T = b.title ? 26 : 6, H = T + b.rows.length * (rowH + 5) + 8, max = b.max ?? Math.max(...b.rows.map(r => r.v));
  const f = v => b.unit === "$" ? "$" + v.toFixed(0) : v.toFixed(b.unit === "M" ? 1 : 0) + " " + b.unit;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
  if (b.title) el("text", { x: 4, y: 15, class: "ttl" }, svg).textContent = b.title;
  b.rows.forEach((r, i) => {
    const y = T + i * (rowH + 5), w = r.v / max * (W - Lw - 60);
    el("text", { x: Lw - 8, y: y + 11, "text-anchor": "end" }, r.run ? svgLink(svg, r.run) : svg).textContent = r.label;
    title(el("rect", { x: Lw, y, width: Math.max(1, w), height: rowH, fill: css(r.color || "--base"), rx: 2 }, svg), `${r.label}: ${f(r.v)}`);
    el("text", { x: Lw + w + 5, y: y + 11 }, svg).textContent = f(r.v);
  });
  host.appendChild(svg);
}

// ---------- comparison tables: baseline vs ablation at the final checkpoint, at 120 min and at matched tokens ----------
const POINTS = [["final", "final"], ["at_120min", "at 120 min"], ["at_matched_tokens", "at equal tokens"]];
function relErr(k, v) { return v == null ? null : kind(k) === "ratio" ? Math.abs(v - 1) : v; }
const TINY = k => kind(k) === "ratio" ? 0.005 : k.endsWith("_mm") ? 0.01 : k === "rgb_mse" ? 0 : 1e-3;  // below this an error counts as exact
function delta(k, a, b) {  // a, b: errors (lower is better)
  if (a == null || b == null) return "";
  const tiny = v => v <= TINY(k);
  if (tiny(a) && tiny(b)) return `<span class="same">both exact</span>`;
  if (tiny(a)) return `<span class="worse">exact → ${kind(k) === "ratio" && Math.abs(b - 1) < 1e-9 ? "dropped" : "off"}</span>`;
  if (tiny(b)) return `<span class="better">→ exact</span>`;
  const r = b / a, txt = r >= 2 || r <= 0.5 ? `×${fmt(r, 2)} error` : `${r > 1 ? "+" : "−"}${Math.abs(100 * (r - 1)).toFixed(0)}% error`;
  return `<span class="${Math.abs(r - 1) < 0.05 ? "same" : r > 1 ? "worse" : "better"}">${txt}</span>`;
}
function compareTable(host, cid) {  // per comparison point: baseline | ablation (with the change in error underneath)
  const c = R.comparisons[cid], val = (r, p, k) => { const i = c.points[p][r]; return i == null ? null : RUNS[r].subs[i].m[k]; };
  const at = (r, p) => { const i = c.points[p][r]; return i == null ? "none yet" : `#${i + 1} · ${RUNS[r].subs[i].t.toFixed(0)} min · ${RUNS[r].subs[i].mtok.toFixed(1)}M tokens`; };
  const card = div(host, "card scroll"), multi = c.base.length > 1;
  card.appendChild(html(`<h3 style="margin-top:0">${c.label}</h3>`));
  let t = `<table class="cmp"><tr><th rowspan="2">metric</th>${POINTS.map(([p, l]) => `<th colspan="2" class="grp">${p === "at_matched_tokens" ? `at ${c.mtok.toFixed(1)}M tokens` : l}</th>`).join("")}</tr><tr>` +
    POINTS.map(() => `<th><span class="sw blk" style="background:${css("--base")}"></span> ${c.base_label}</th><th><span class="sw blk" style="background:${css("--abl")}"></span> ${c.abl_label}</th>`).join("") + "</tr>";
  t += `<tr class="meta"><td>submission used (click to open the run)</td>${POINTS.map(([p]) => `<td title="${c.base.map(r => name(r) + ": " + at(r, p)).join("\n")}">${multi ? c.base.map((r, i) => runLink(r, "run " + "ABCDEF"[i])).join(" · ") : runLink(c.base[0], at(c.base[0], p))}</td><td>${runLink(c.abl, at(c.abl, p))}</td>`).join("")}</tr>`;
  for (const k of c.keys) {
    t += `<tr><td>${label(k)}${kind(k) === "ratio" ? " ÷ true" : ""}</td>`;
    for (const [p] of POINTS) {
      const bv = c.base.map(r => val(r, p, k)).filter(v => v != null), av = val(c.abl, p, k);
      const bm = bv.length ? bv.reduce((a, b) => a + b, 0) / bv.length : null;
      const range = multi && bv.length > 1 ? `<span class="sub">range ${fmtKey(k, Math.min(...bv))}–${fmtKey(k, Math.max(...bv))}</span>` : "";
      const inBand = multi && bv.length > 1 && av != null && av >= Math.min(...bv) && av <= Math.max(...bv);
      const errs = bv.map(v => relErr(k, v)), be = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : null;
      t += `<td>${fmtKey(k, bm)}${range}</td><td class="ab">${fmtKey(k, av)}<span class="sub delta">${inBand ? `<span class="same">within run-to-run range</span>` : delta(k, be, relErr(k, av))}</span></td>`;
    }
    t += "</tr>";
  }
  card.appendChild(html(t + "</table>"));
  card.appendChild(html(`<p class="note">"÷ true" rows: the agent's estimate divided by the true value (×1 = exact). Under each value of the changed condition: how its error compares with the baseline${multi ? " mean" : ""}. "Submission used" = which submission each column reads (number · minute · tokens so far)${multi ? `; hover "runs" for each baseline repeat` : ""}.</p>`));
}

// ---------- estimate-ratio dot plot ----------
function ratios(host, b) {
  const lo = 0.02, hi = 3.5, W = 760, Lw = 250, sx = v => Lw + (Math.log(Math.max(v, lo)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (W - Lw - 110);
  for (const p of b.panels) {
    const card = div(host, "card"); card.appendChild(html(`<h3 style="margin-top:0">${p.title}</h3>`));
    legend(div(card, "legend"), p.keys.map(k => [label(k), KEYCOLOR[k], "blk"]).concat([["0 = dropped (hollow, at left edge)", null], ["exact", null, "dash"]]));
    const rowH = 22, T = 8, H = T + p.rows.length * rowH + 26, svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
    for (const v of [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]) { el("line", { x1: sx(v), x2: sx(v), y1: T - 4, y2: H - 22, class: v === 1 ? "truth" : "gr" }, svg); el("text", { x: sx(v), y: H - 6, "text-anchor": "middle" }, svg).textContent = v === 1 ? "exact" : "×" + v; }
    p.rows.forEach((row, i) => {
      const y = T + i * rowH + 10;
      if (row.header) { el("text", { x: Lw - 10, y: y + 4, "text-anchor": "end", style: "font-weight:600;fill:var(--fg)" }, svg).textContent = row.header; return; }
      const [r, lab] = row;
      el("text", { x: Lw - 10, y: y + 4, "text-anchor": "end" }, svgLink(svg, r)).textContent = lab;
      const vals = p.keys.map(k => [k, RUNS[r].subs[RUNS[r].subs.length - 1].m[k]]).filter(([, v]) => v != null);
      vals.forEach(([k, v], j) => {
        const off = (j - (vals.length - 1) / 2) * 2.2, zero = v === 0;
        title(el("circle", { cx: sx(v), cy: y + off, r: 5.5, fill: zero ? css("--card") : css(KEYCOLOR[k]), stroke: zero ? css(KEYCOLOR[k]) : css("--card"), "stroke-width": zero ? 2 : 1.5, opacity: 0.92 }, svg),
          `${name(r)} · ${label(k)}: ${xr(v)} (estimate ${fmt(RUNS[r].subs[RUNS[r].subs.length - 1].est[k.slice(2)], 4)}, true ${fmt(RUNS[r].truth[k.slice(2)], 4)})`);
      });
      const nz = vals.map(([, v]) => v).filter(v => v > 0), g = nz.length ? Math.exp(nz.reduce((a, v) => a + Math.log(v), 0) / nz.length) : null;
      const common = nz.length > 1 && nz.every(v => Math.abs(v / g - 1) < 0.03);
      el("text", { x: W - 104, y: y + 4, style: "fill:var(--fg)" }, svg).textContent = !g ? "" : Math.abs(g - 1) < 2e-3 && common ? "exact" : common ? `all ${xr(g)}` : `${xr(Math.min(...nz))}–${fmt(Math.max(...nz), 3)}`;
    });
    card.appendChild(svg);
  }
}

// ---------- per-submission convergence small multiples ----------
function groupCurve(runs, k) {
  // Several runs of one condition: at each submission time (once every run has submitted), the mean and range of each run's latest value.
  const subs = runs.map(r => RUNS[r].subs.filter(x => x.m[k] != null));
  if (subs.some(a => !a.length)) return { pts: [], spread: [] };
  const start = Math.max(...subs.map(a => a[0].t));
  const times = [...new Set(subs.flatMap(a => a.map(x => x.t)).filter(t => t >= start))].sort((a, b) => a - b);
  const latest = (a, t) => a.filter(x => x.t <= t).pop().m[k];
  const rows = times.map(t => [t, subs.map(a => latest(a, t))]);
  return { pts: rows.map(([t, v]) => [t, v.reduce((x, y) => x + y, 0) / v.length]), spread: rows.map(([t, v]) => [t, Math.min(...v), Math.max(...v)]) };
}
function convergence(host, b) {
  const card = div(host, "card");
  if (b.title) card.appendChild(html(`<h3 style="margin-top:0">${b.title}</h3>`));
  const series = b.series.map((s, i) => ({ ...s, color: s.color || PALETTE[i], label: s.runs ? `${s.label} (mean of ${s.runs.length} runs)` : s.label }));
  legend(div(card, "legend"), series.filter(s => !s.faded).map(s => [s.label, s.color]).concat(series.some(s => s.faded) ? [["faded: extended +2 h", null]] : [], [["true value (ratio plots)", null, "dash"]]));
  const grid = div(card, "minis");
  for (const k of b.keys) {
    const ss = series.map(s => ({ name: s.label, color: s.color, dash: s.dash, faded: s.faded,
      ...(s.runs ? groupCurve(s.runs, k) : { pts: RUNS[s.run].subs.map(x => [x.t, x.m[k]]) }) }));
    if (!ss.some(s => s.pts.some(p => p[1] != null))) continue;
    const ratio = kind(k) === "ratio";
    lineChart(div(grid), { title: label(k) + (ratio ? " ÷ true" : ""), series: ss, log: true, ratio, truth: ratio ? 1 : null, xmax: b.xmax, at120: true,
      floor: ratio ? 0.01 : k.endsWith("_mm") ? 0.01 : 0, fmtv: v => fmtKey(k, v) });
  }
  card.appendChild(html(`<p class="note">One point per submission at its submission time; log scale.${series.some(s => s.runs) ? " Conditions with several runs: line = mean of the runs' latest values, shaded = their range (min to max; with three runs the range is more honest than a standard deviation), drawn from when every run has submitted." : ""} Hollow points at the bottom of a ratio plot: the parameter was left out (0). Filled points on the floor labeled \"exact\" of an error plot: error below 0.01 mm. The dotted vertical line is the 120-minute mark.</p>`));
}

// ---------- strategy taxonomy (all runs in R.order) ----------
const SCRIPT_CATS = [["fit_sim", "fit parameters by optimizer or sweep over local MuJoCo simulations", "--base"], ["fit_data", "fit by optimizer or sweep on images or recorded data", "--fourth"],
  ["simulate", "single local simulation or render of a candidate (check)", "--third"], ["analytic", "closed-form physics or linear algebra", "--abl"], ["image", "image processing only", "--s5"], ["other", "file and project housekeeping", "--faint"]];
const NOTE_CATS = [["image", "Image measurement"], ["analytic", "Analytic physics"], ["fitting", "Systematic fitting"], ["adjust_check", "Guess–adjust–check"], ["prior", "Priors / defaults / simplicity"]];
function strategy(host) {
  const order = R.order, n = order.length;
  host.appendChild(html(`<p class="answer">${R.strategy.answer}</p>`));
  const timeCats = [["bash", "bash scripts", "--base"], ["view_image", "viewing images", "--s5"], ["ref", "actions on the hidden scene", "--abl"], ["rec", "actions on the agent's copy", "--third"], ["submit", "submit / finish", "--fourth"]];
  const timeRow = r => { const s = RUNS[r].call_seconds, v = { bash: s.bash || 0, view_image: s.view_image || 0, ref: 0, rec: 0, submit: (s.submit || 0) + (s.finish || 0) };
    for (const k of ["observe", "grasp", "move", "release", "reset", "arm_target"]) v.ref += s[k] || 0; v.rec += s.load_reconstruction || 0;
    // controller actions name the world in their arguments only; split physical-action time by the recorded per-world action counts
    const a = RUNS[r].actions, cnt = suf => Object.entries(a).filter(([k]) => k.endsWith(suf) && !/submit|finish|load_/.test(k)).reduce((x, [, m]) => x + m, 0);
    const nref = cnt("|reference"), nrec = cnt("|reconstruction"), phys = v.ref;
    v.ref = phys * nref / Math.max(1, nref + nrec); v.rec += phys * nrec / Math.max(1, nref + nrec); return v; };
  const sum = rows => { const out = {}; for (const r of rows) for (const [k, v] of Object.entries(r)) out[k] = (out[k] || 0) + v; return out; };
  let c = div(host, "card"); c.appendChild(html(`<h3 style="margin-top:0">Where the time goes: wall time between model turns, by what the turn called</h3>`));
  legend(div(c, "legend"), timeCats.map(x => [x[1], x[2], "blk"]));
  stackedBars(c, [{ label: `all ${n} runs`, bold: true, values: sum(order.map(timeRow)) }, ...order.map(r => ({ label: name(r), run: r, values: timeRow(r) }))], timeCats, { unit: "h" });
  c.appendChild(html(`<p class="note">Approximate: the time from one model reply to the next is attributed to the tools that reply called (split evenly). The agent's own thinking time is not separated out.</p>`));
  c = div(host, "card"); c.appendChild(html(`<h3 style="margin-top:0">What their scripts compute: every bash call, classified</h3>`));
  legend(div(c, "legend"), SCRIPT_CATS.map(x => [x[1], x[2], "blk"]));
  const sall = sum(order.map(r => RUNS[r].taxonomy.script_counts)), tot = Object.values(sall).reduce((a, b) => a + b, 0), sv = R.taxonomy_validation.scripts;
  stackedBars(c, [{ label: `all ${n} runs`, bold: true, values: sall }, ...order.map(r => ({ label: name(r), run: r, values: RUNS[r].taxonomy.script_counts }))], SCRIPT_CATS);
  c.appendChild(html(`<p class="note">${tot} bash calls. One kind per call, first rule that applies, in legend order. A hand check of ${sv.n} random calls agreed on ${sv.agree} (${(100 * sv.agree / sv.n).toFixed(0)}%).</p>`));
  c = div(host, "card scroll"); c.appendChild(html(`<h3 style="margin-top:0">What their progress notes say they are doing: share of notes mentioning each strategy</h3>`));
  const totals = Object.fromEntries(NOTE_CATS.map(([k]) => [k, 0])); let ntot = 0, app = 0;
  for (const r of order) { ntot += RUNS[r].taxonomy.notes; app += RUNS[r].taxonomy.appearance; for (const [k] of NOTE_CATS) totals[k] += RUNS[r].taxonomy.note_counts[k] || 0; }
  const cell = (m, d) => `<td style="background:color-mix(in srgb, ${css("--base")} ${Math.round(70 * m / d)}%, transparent)">${Math.round(100 * m / d)}%</td>`;
  c.appendChild(html(`<table class="heat"><tr><th>scene · condition</th><th>notes</th>${NOTE_CATS.map(x => `<th>${x[1]}</th>`).join("")}<th>about appearance</th></tr>` +
    `<tr style="font-weight:600"><td>all ${n} runs</td><td>${ntot}</td>${NOTE_CATS.map(([k]) => cell(totals[k], ntot)).join("")}${cell(app, ntot)}</tr>` +
    order.map(r => { const t = RUNS[r].taxonomy; return `<tr><td>${runLink(r)}</td><td>${t.notes}</td>${NOTE_CATS.map(([k]) => cell(t.note_counts[k] || 0, t.notes)).join("")}${cell(t.appearance, t.notes)}</tr>`; }).join("") + "</table>"));
  const NV = R.taxonomy_validation.notes;
  c.appendChild(html(`<p class="note">A note can carry several labels. Agreement of the keyword rules with hand labels on ${NV.n} held-out notes (11-03 runs): ` +
    NOTE_CATS.concat([["appearance", "appearance"]]).map(([k, l]) => `${l} ${Math.round(100 * NV.categories[k].agreement)}% (κ ${NV.categories[k].kappa.toFixed(2)})`).join(", ") +
    `. Analytic physics is rarely stated in notes (the rule missed most hand-labeled cases), so the script classification above is the better measure of it.</p>`));
  host.appendChild(html(`<h3>Examples, verbatim</h3>`));
  const g = div(host, "grid2");
  g.innerHTML = Object.entries(R.examples).map(([cat, items]) => `<div class="card"><b>${cat}</b>${items.map(e => `<div class="quote">“${e.text}”<span class="src">${runLink(e.run, e.name)} · minute ${e.minute} · ${e.utc.slice(0, 16).replace("T", " ")} UTC</span></div>`).join("")}</div>`).join("");
}

// ---------- traceability table ----------
function runsTable(host) {
  const c = div(host, "card scroll");
  c.appendChild(html(`<table><tr><th>scene · condition</th><th>minutes</th><th>submissions</th><th>model tokens (M)</th><th>Chamfer (mm)</th><th>COM error (mm)</th><th>damping error</th><th>dry-friction error</th><th>image MSE</th><th>cost ($)</th></tr>` +
    R.order.map(r => { const x = RUNS[r], m = x.subs[x.subs.length - 1].m; return `<tr><td>${runLink(r)}<div class="ids">${r}</div></td><td>${x.wall_min.toFixed(0)}</td><td>${x.submissions}</td><td>${x.final_mtok.toFixed(1)}</td><td>${mmf(m.chamfer_mm)}</td><td>${mmf(m.com_mm)}</td><td>${pct(m.damping_rel)}</td><td>${pct(m.frictionloss_rel)}</td><td>${fmt(m.rgb_mse, 2)}</td><td>${x.cost_usd.toFixed(2)}</td></tr>`; }).join("") +
    `<tr><td colspan="10" class="note">Second line: internal run ID, for tracing only. Values are from each run's last submission; errors are the private evaluator's scores.</td></tr></table>`));
}

// ---------- 3D overlay ----------
function viewer(host) {
  const runs = Object.keys(R.viewer).filter(r => R.viewer[r].group === host.dataset.group);
  host.innerHTML = `<div class="vctl"><select>${runs.map(r => `<option value="${r}">${name(r)}</option>`).join("")}</select>
    <label><input type="checkbox" checked data-k="truth"> truth (solid)</label><label><input type="checkbox" checked data-k="recon"> reconstruction (wireframe)</label>
    <label><input type="checkbox" data-k="colors"> real colors</label><button>reset view</button><a class="openrun">open this run</a></div>
    <canvas></canvas><div class="vctl"><span>submission</span><input type="range" min="0" value="0"><span class="vread"></span></div>
    <div class="legend"><span><i class="sw blk" style="background:#8a96a3"></i>truth</span><span><i class="sw blk" style="background:#eb6834"></i>reconstruction</span><span>● true COM&nbsp;&nbsp;◆ reconstructed COM</span><span>drag to orbit, scroll or pinch to zoom</span></div>`;
  const canvas = host.querySelector("canvas"), sel = host.querySelector("select"), slider = host.querySelector("input[type=range]"), read = host.querySelector(".vread");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(35, 1, 0.005, 50);
  camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.9)); const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(2, -3, 4); scene.add(sun);
  const controls = new THREE.OrbitControls(camera, canvas);
  const truthG = new THREE.Group(), reconG = new THREE.Group(); scene.add(truthG, reconG);
  const grid = new THREE.GridHelper(2, 20, 0x888888, 0x888888); grid.rotation.x = Math.PI / 2; grid.material.opacity = 0.25; grid.material.transparent = true; scene.add(grid);
  const geometry = (g, lo) => {
    const s = g.size, u = lo ? 14 : 28, v = lo ? 9 : 18; let geo;
    if (g.type === "sphere") geo = new THREE.SphereGeometry(s[0], u, v);
    else if (g.type === "ellipsoid") { geo = new THREE.SphereGeometry(1, u, v); geo.scale(s[0], s[1], s[2]); }
    else if (g.type === "box") geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    else if (g.type === "capsule") { geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], lo ? 4 : 8, u); geo.rotateX(Math.PI / 2); }
    else if (g.type === "cylinder") { geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], u); geo.rotateX(Math.PI / 2); }
    else if (g.type === "mesh") { geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.Float32BufferAttribute(g.mesh.vert.flat(), 3)); geo.setIndex(g.mesh.face.flat()); geo.computeVertexNormals(); }
    else throw new Error("unsupported geom type " + g.type);
    return geo;
  };
  const place = (mesh, g) => { mesh.position.set(...g.pos); mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]); return mesh; };
  const colorOf = (g, fallback, real) => real ? new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]) : new THREE.Color(fallback);
  const clear = grp => { while (grp.children.length) { const c = grp.children.pop(); c.geometry.dispose(); c.material.dispose(); } };
  let scale = 0.02;
  const comMarker = (pos, color, cross) => { const m = new THREE.Mesh(cross ? new THREE.OctahedronGeometry(1.25 * scale) : new THREE.SphereGeometry(scale, 12, 8), new THREE.MeshBasicMaterial({ color, depthTest: false })); m.position.set(...pos); m.renderOrder = 10; return m; };
  const flag = k => host.querySelector(`[data-k=${k}]`).checked;
  function draw() {
    const v = R.viewer[sel.value], sub = v.subs[+slider.value], real = flag("colors");
    clear(truthG); clear(reconG);
    for (const g of v.source.geoms) truthG.add(place(new THREE.Mesh(geometry(g), new THREE.MeshStandardMaterial({ color: colorOf(g, 0x8a96a3, real), transparent: true, opacity: 0.45, depthWrite: false, roughness: 0.8 })), g));
    truthG.add(comMarker(v.source.com, 0x1b1d1f));
    for (const g of sub.geoms) reconG.add(place(new THREE.Mesh(geometry(g, true), new THREE.MeshBasicMaterial({ color: colorOf(g, 0xeb6834, real), wireframe: true, transparent: true, opacity: 0.8 })), g));
    reconG.add(comMarker(sub.com, 0xeb6834, true));
    truthG.visible = flag("truth"); reconG.visible = flag("recon");
    read.textContent = `${sub.label} of ${v.subs.length} · ${sub.t.toFixed(0)} min · Chamfer ${sub.chamfer == null ? "—" : mmf(sub.chamfer) + " mm"} · COM error ${sub.com_err == null ? "—" : mmf(sub.com_err) + " mm"}`;
  }
  function frame() {
    const v = R.viewer[sel.value], box = new THREE.Box3();
    for (const g of v.source.geoms) box.expandByPoint(new THREE.Vector3(...g.pos));
    const c = box.getCenter(new THREE.Vector3()), r = Math.max(0.12, box.getSize(new THREE.Vector3()).length());
    scale = Math.max(0.004, r / 30);
    controls.target.copy(c); camera.position.set(c.x + 1.3 * r, c.y - 1.8 * r, c.z + 1.0 * r); grid.position.set(c.x, c.y, 0); controls.update();
  }
  function pick() { host.querySelector(".openrun").href = runHref(sel.value); const v = R.viewer[sel.value]; slider.max = v.subs.length - 1; slider.value = v.subs.length - 1; frame(); draw(); }
  sel.onchange = pick; slider.oninput = draw; host.querySelectorAll("input[type=checkbox]").forEach(c => c.onchange = draw); host.querySelector("button").onclick = frame;
  function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (canvas.width !== Math.round(w * renderer.getPixelRatio())) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); } }
  pick();
  (function loop() { resize(); controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
}

// ---------- page assembly ----------
const BLOCKS = {
  html: (h, b, key) => h.appendChild(html(`<div data-edit="${key}">${b.html}</div>`)),
  compare: (h, b) => b.ids.forEach(id => compareTable(h, id)),
  ratios: (h, b) => { if (b.title) h.appendChild(html(`<p class="note">${b.title}</p>`)); ratios(h, b); },
  convergence: convergence,
  table: (h, b) => { const c = div(h, "card scroll"); c.innerHTML = (b.title ? `<h3 style="margin-top:0">${b.title}</h3>` : "") +
    `<table><tr>${b.head.map(x => `<th>${x}</th>`).join("")}</tr>${b.rows.map(r => `<tr>${r.map(x => `<td>${x}</td>`).join("")}</tr>`).join("")}</table>` + (b.note ? `<p class="note">${b.note}</p>` : ""); },
  quotes: (h, b) => { const c = div(h, "card"); c.innerHTML = `<b>${b.title}</b>` + b.items.map(e => `<div class="quote">“${e.text}”<span class="src">${runLink(e.run, e.name)} · minute ${e.minute}</span></div>`).join(""); },
  bars: (h, b) => hbars(div(h, "card"), b),
  viewer: (h, b) => { const c = div(h, "card"); c.innerHTML = `<h3 style="margin-top:0">${b.title}</h3><div class="viewer" data-group="${b.group}"></div>`; },
  videos: (h, b) => { const g = div(h, "grid2"); for (const r of b.runs) { const v = R.videos.find(x => x.run === r);
    g.appendChild(html(`<div class="card vid"><video controls loop muted playsinline preload="metadata" src="videos/${v.video}"></video><p><b>${runLink(r)}</b></p><p class="muted">the same ${v.actions} actions in both (${v.actions_listed.join(", ")}) · agent's scene from its load #${v.load}</p></div>`)); }
    if (b.note) h.appendChild(html(`<p class="note">${b.note}</p>`)); },
  strategy: (h) => strategy(h),
  runs: (h) => runsTable(h),
};
// ---------- text edits: edits.js holds {key: html} saved from edit mode; applied to R before rendering ----------
function applyEdits(E) {
  for (const [key, v] of Object.entries(E)) {
    const p = key.split(":");
    if (["title", "subtitle", "intro"].includes(key)) R.meta[key] = v;
    else if (p[0] === "glossary") R.glossary[+p[1]][+p[2]] = v;
    else if (p[0] === "methods") R.methods[+p[1]] = v;
    else if (p[0] === "section") {
      const s = R.sections.find(x => x.id === p[1]);
      if (!s) throw new Error(`saved edit ${key}: no section ${p[1]}`);
      if (p[2] === "block") { if (s.blocks[+p[3]]?.type !== "html") throw new Error(`saved edit ${key}: not a text block`); s.blocks[+p[3]].html = v; }
      else s[p[2]] = v;
    } else throw new Error(`unknown saved edit key ${key}`);
  }
}
function editMode() {  // contenteditable on every [data-edit]; Save posts the changed ones to the report server (serve-report.py)
  const bar = document.body.appendChild(html(`<div class="editbar"><button data-a="edit">Edit text</button></div>`));
  let before = null;
  const nodes = () => [...document.querySelectorAll("[data-edit]")];
  bar.onclick = async e => {
    const a = e.target.dataset.a;
    if (a === "edit") {
      before = new Map(nodes().map(n => [n.dataset.edit, n.innerHTML]));
      nodes().forEach(n => { n.contentEditable = "true"; });
      document.body.classList.add("editing");
      bar.innerHTML = `<span>Editing: links are not followed while editing</span><button data-a="save">Save</button><button data-a="discard">Discard</button>`;
    } else if (a === "discard") location.reload();
    else if (a === "save") {
      const changed = Object.fromEntries(nodes().filter(n => n.innerHTML !== before.get(n.dataset.edit)).map(n => [n.dataset.edit, n.innerHTML]));
      const res = await fetch("edits", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...window.EDITS, ...changed }) })
        .catch(err => ({ ok: false, statusText: String(err) }));
      if (!res.ok) { alert(`Not saved (${res.status || ""} ${res.statusText}). Saving needs the report server (serve-report.py); your edits are still on the page.`); return; }
      location.reload();
    }
  };
}
(function page() {
  document.title = "sim2sim Results";
  if (typeof window.EDITS !== "object") throw new Error("edits.js did not load");
  applyEdits(window.EDITS);
  $("title").innerHTML = R.meta.title; $("title").dataset.edit = "title";
  $("subtitle").innerHTML = R.meta.subtitle.replace("{n}", R.order.length); $("subtitle").dataset.edit = "subtitle";
  const tocList = list => "<ul>" + list.map(s => `<li><a href="#${s.id}">${s.title}</a></li>`).join("") + "</ul>";
  $("toc").innerHTML = `<b>Results</b>${tocList(R.sections.filter(s => !s.appendix))}<b>Appendix</b>${tocList(R.sections.filter(s => s.appendix))}`;
  $("intro").innerHTML = R.meta.intro; $("intro").dataset.edit = "intro";
  $("glossary").innerHTML = R.glossary.map(([t, d], i) => `<div><b data-edit="glossary:${i}:0">${t}</b><span data-edit="glossary:${i}:1">${d}</span></div>`).join("");
  const host = $("sections");
  for (const s of R.sections) {
    host.appendChild(html(`<h2 id="${s.id}"><span data-edit="section:${s.id}:title">${s.title}</span>${s.tag ? ` <span class="tag">${s.tag}</span>` : ""}</h2>`));
    if (s.q) host.appendChild(html(`<p class="q" data-edit="section:${s.id}:q">${s.q}</p>`));
    if (s.answer) host.appendChild(html(`<p class="answer" data-edit="section:${s.id}:answer">${s.answer}</p>`));
    s.blocks.forEach((b, i) => BLOCKS[b.type](host, b, `section:${s.id}:block:${i}`));
  }
  $("methods-list").innerHTML = R.methods.map((m, i) => `<li data-edit="methods:${i}">${m}</li>`).join("");
  editMode();
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting && !e.target.dataset.ready) { e.target.dataset.ready = 1; viewer(e.target); } }), { rootMargin: "200px" });
  document.querySelectorAll(".viewer").forEach(v => io.observe(v));
})();
