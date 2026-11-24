// Per-run page: outcome, scores at every submission, and the full model conversation (runs/RUN_ID/chat.js, from report-chat-extract.py).
"use strict";
const RUN = new URLSearchParams(location.search).get("run"), X = RUNS[RUN];
const minute = t => (t / 60).toFixed(1);
const LONG = 1500;  // characters; longer text parts start collapsed
const textBlock = s => s.length > LONG
  ? `<details class="long"><summary><span class="pre">${esc(s.slice(0, 400))}</span> <i>… ${s.length.toLocaleString()} characters, click to expand</i></summary><div class="pre">${esc(s)}</div></details>`
  : `<div class="pre">${esc(s)}</div>`;

function header() {
  document.title = `${name(RUN)} · sim2sim`;
  $("title").textContent = name(RUN);
  $("subtitle").innerHTML = `Run <code>${RUN}</code> · <code>gpt-6-astra</code>, medium reasoning, strict sandbox`;
  const end = X.reason === "budget_suspended" ? "stopped by its API cost limit" : X.reason === "agent_finish" ? "the agent called finish" : esc(X.reason);
  $("facts").innerHTML = [["ended", end], ["wall clock", `${X.wall_min.toFixed(0)} min`], ["submissions", X.submissions],
    ["model tokens", `${X.final_mtok.toFixed(1)}M`], ["API cost", `$${X.cost_usd.toFixed(2)}`]]
    .map(([k, v]) => `<div class="card"><div class="small muted">${k}</div><div class="big">${v}</div></div>`).join("");
  const keys = Object.keys(X.subs[0].m).filter(k => X.subs.some(s => s.m[k] != null));
  $("scores").innerHTML = `<table><tr><th>#</th><th>minute</th><th>tokens so far</th>${keys.map(k => `<th>${label(k)}${kind(k) === "ratio" ? " ÷ true" : ""}</th>`).join("")}</tr>` +
    X.subs.map((s, i) => `<tr id="sub-${i + 1}"><td>${i + 1}</td><td>${s.t.toFixed(0)}</td><td>${s.mtok.toFixed(1)}M</td>${keys.map(k => `<td>${fmtKey(k, s.m[k])}</td>`).join("")}</tr>`).join("") + "</table>";
}

function parts(content, img) {  // message content: a string, or text/image_url parts
  if (typeof content === "string") return textBlock(content);
  return content.map(p => p.type === "text" ? textBlock(p.text) : p.type === "image_url" ? `<img class="obs" loading="lazy" src="${img}${p.image_url.url}" alt="image given to the model">`
    : `<div class="pre">${esc(JSON.stringify(p))}</div>`).join("");
}

function toolCall(c) {
  let args;
  try { args = JSON.parse(c.arguments); } catch {  // the model's own malformed arguments are data: show them as sent
    return `<div class="call"><span class="role tool">${esc(c.name)} (arguments are not valid JSON)</span>${textBlock(c.arguments)}</div>`;
  }
  const body = c.name === "bash" ? `<div class="pre code">${esc(args.command)}</div>` + (Object.keys(args).length > 1 ? `<div class="small muted">${esc(JSON.stringify({ ...args, command: undefined }))}</div>` : "")
    : textBlock(JSON.stringify(args, null, 1));
  return `<div class="call"><span class="role tool">${esc(c.name)}</span>${body}</div>`;
}

function mark(m) {
  if (m.kind === "termination") return `<div class="mark">min ${minute(m.t)} · run ended (${esc(m.reason)}) after ${m.submissions} submissions</div>`;
  const s = X.subs[m.submission - 1], keys = Object.keys(s.m).filter(k => s.m[k] != null && (k.startsWith("r:") || k.endsWith("_mm")));
  return `<div class="mark" id="mark-${m.submission}">min ${minute(m.t)} · <a href="#sub-${m.submission}">submission ${m.submission}</a> (${esc(m.status)}) · private scores: ` +
    keys.map(k => `${label(k)}${kind(k) === "ratio" ? " ÷ true" : ""} ${fmtKey(k, s.m[k])}`).join(", ") + "</div>";
}

function render() {
  const C = window.CHAT, img = `runs/${RUN}/`, toolName = {};
  if (C.run !== RUN) throw new Error(`conversation file is for ${C.run}, not ${RUN}`);
  const nimg = C.turns.reduce((a, t) => a + t.new.reduce((b, m) => b + (Array.isArray(m.content) ? m.content.filter(p => p.type === "image_url").length : 0), 0), 0);
  $("chatctl").innerHTML = `<span>${C.turns.length} model calls · ${nimg} images given to the model</span>` +
    `<label><input type="checkbox" id="tog-tool" checked> tool results</label><label><input type="checkbox" id="tog-img" checked> images</label>` +
    C.marks.filter(m => m.kind === "submission").map(m => `<a href="#mark-${m.submission}">submission ${m.submission}</a>`).join(" ");
  const marks = [...C.marks], out = [];
  C.turns.forEach((turn, i) => {
    while (marks.length && marks[0].t <= turn.t) out.push(mark(marks.shift()));
    let h = `<div class="turn"><div class="turnhead">model call ${i + 1} · min ${minute(turn.t)} · ${turn.context} messages in context` +
      (turn.compaction ? ` · <b>context-summary request</b>: the harness keeps only the reply's text, replaces older turns with it, and runs none of its tool calls` : "") + `</div>`;
    for (const msg of turn.new) {
      const who = msg.role === "tool" ? `result of ${esc(toolName[msg.tool_call_id] || "tool")}` : msg.role;
      h += `<div class="msg ${msg.role}"><span class="role ${msg.role}">${who}</span>${parts(msg.content, img)}</div>`;
    }
    const r = turn.reply;
    if (!r) h += `<div class="msg assistant"><span class="role assistant">model</span><i class="muted">no reply recorded (the run ended during this call)</i></div>`;
    else {
      (r.tool_calls || []).forEach(c => { toolName[c.id] = c.name; });
      const extra = Object.keys(r).filter(k => !["t", "content", "tool_calls", "usage"].includes(k));
      h += `<div class="msg assistant"><span class="role assistant">model · min ${minute(r.t)}</span>${r.content ? textBlock(r.content) : ""}` +
        extra.map(k => `<details><summary>provider-returned ${esc(k)}</summary>${textBlock(typeof r[k] === "string" ? r[k] : JSON.stringify(r[k], null, 1))}</details>`).join("") +
        (r.tool_calls || []).map(toolCall).join("") + "</div>";
    }
    out.push(h + "</div>");
  });
  marks.forEach(m => out.push(mark(m)));
  $("chat").innerHTML = out.join("");
  const tog = (id, cls) => { $(id).onchange = e => document.body.classList.toggle(cls, !e.target.checked); };
  tog("tog-tool", "hide-tool"); tog("tog-img", "hide-img");
}

// ---------- recording player: runs/RUN/recording.js (window.RECORDING) and one MP4 per world, from run-video.py ----------
function recording() {
  const D = window.RECORDING, host = $("recording"), worlds = Object.keys(D.worlds);
  const physical = a => a && !["progress", "inference"].includes(a.op);  // notes and model calls are logged as actions too
  if (D.run !== RUN) throw new Error(`recording is for ${D.run}, not ${RUN}`);
  const end = Math.max(...worlds.map(w => D.worlds[w].t.at(-1)), ...D.notes.map(n => n.t), ...D.actions.map(a => a.end || a.start));
  const names = { reference: "hidden scene", reconstruction: "agent's reconstruction" };
  host.innerHTML = `<div class="rec-panes">${worlds.map(w => `<figure><video id="v-${w}" muted playsinline preload="auto" src="runs/${RUN}/recording/${D.worlds[w].file}"></video>` +
      `<figcaption><b>${names[w] || w}</b> · <span id="i-${w}"></span></figcaption></figure>`).join("")}</div>` +
    `<div class="rec-ctl"><button id="rec-play">Play</button><select id="rec-speed"><option value="5">5 frames/s</option><option value="10" selected>10 frames/s</option>` +
    `<option value="30">30 frames/s</option></select><input type="range" id="rec-t" min="0" max="${end}" step="any" value="0"><span id="rec-clock"></span></div>` +
    `<svg id="rec-marks" viewBox="0 0 1000 34" preserveAspectRatio="none"></svg><div class="rec-now"><span id="rec-action"></span><div id="rec-note" class="muted"></div></div>`;
  const svg = $("rec-marks"), x = t => 1000 * t / end;
  const row = { reference: 4, reconstruction: 13 };
  for (const a of D.actions) if (a.start != null && physical(a) && !["submit", "finish", "load_reconstruction"].includes(a.op))
    title(el("rect", { x: x(a.start), y: row[a.world] ?? 4, width: Math.max(1.5, x((a.end ?? a.start) - a.start)), height: 7, fill: css(a.world === "reference" ? "--base" : "--third") }, svg),
          `${a.world} · ${a.op} at min ${(a.start / 60).toFixed(1)}`);
  for (const s of D.submissions) title(el("path", { d: `M${x(s.t)},22 l4,8 h-8 z`, fill: css("--abl") }, svg), `submission at min ${(s.t / 60).toFixed(1)}`);
  for (const n of D.notes) title(el("circle", { cx: x(n.t), cy: 31, r: 1.6, fill: css("--faint") }, svg), `min ${(n.t / 60).toFixed(1)}: ${n.text.slice(0, 200)}`);
  const latest = (ts, t) => { let lo = 0, hi = ts.length - 1, best = -1; while (lo <= hi) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1; } return best; };
  const events = [...new Set(worlds.flatMap(w => D.worlds[w].t))].sort((a, b) => a - b);
  function show(t) {
    $("rec-t").value = t;
    $("rec-clock").textContent = `min ${(t / 60).toFixed(1)} of ${(end / 60).toFixed(0)}`;
    for (const w of worlds) {
      const W = D.worlds[w], i = latest(W.t, t), v = $("v-" + w);
      v.style.opacity = i < 0 ? 0.25 : 1;
      if (i >= 0 && Math.abs(v.currentTime - (i + 0.5) / D.fps) > 0.02) v.currentTime = (i + 0.5) / D.fps;
      const a = i >= 0 && W.action[i] != null && physical(D.actions[W.action[i]]) ? D.actions[W.action[i]] : null;
      $("i-" + w).textContent = i < 0 ? "no frame yet" : `frame ${i + 1} of ${W.t.length}` + (a ? ` · ${a.op}` : "") + (W.sim_time[i] != null ? ` · sim t ${(+W.sim_time[i]).toFixed(2)} s` : "");
    }
    const act = [...D.actions].reverse().find(a => physical(a) && a.start != null && a.start <= t);
    $("rec-action").textContent = act ? `last action: ${act.world} · ${act.op} (min ${(act.start / 60).toFixed(1)})` : "";
    const note = [...D.notes].reverse().find(n => n.t <= t);
    $("rec-note").textContent = note ? `note at min ${(note.t / 60).toFixed(1)}: ${note.text}` : "";
  }
  let playing = null;
  $("rec-t").oninput = e => show(+e.target.value);
  svg.onclick = e => { const r = svg.getBoundingClientRect(); show(end * (e.clientX - r.left) / r.width); };
  $("rec-play").onclick = () => {
    if (playing) { clearInterval(playing); playing = null; $("rec-play").textContent = "Play"; return; }
    $("rec-play").textContent = "Pause";
    let k = Math.max(0, latest(events, +$("rec-t").value) + 1);
    playing = setInterval(() => { if (k >= events.length) { $("rec-play").click(); return; } show(events[k++]); }, 1000 / +$("rec-speed").value);
  };
  show(events[0] ?? 0);
}
if (!X) { document.body.innerHTML = `<main><p>Unknown run <code>${esc(RUN)}</code>. <a href="index.html#runs">All runs</a></p></main>`; throw new Error("unknown run " + RUN); }
header();
const script = document.createElement("script");
script.src = `runs/${RUN}/chat.js`;
script.onload = render;
script.onerror = () => { $("chat").innerHTML = `<p>Conversation file <code>runs/${esc(RUN)}/chat.js</code> is missing from this copy of the report.</p>`; };
document.body.appendChild(script);
const rec = document.createElement("script");
rec.src = `runs/${RUN}/recording/recording.js`;
rec.onload = recording;
rec.onerror = () => { $("recording").innerHTML = `<p class="muted">No recording in this copy of the report (<code>runs/${esc(RUN)}/recording/</code>).</p>`; };
document.body.appendChild(rec);
