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

if (!X) { document.body.innerHTML = `<main><p>Unknown run <code>${esc(RUN)}</code>. <a href="index.html#runs">All runs</a></p></main>`; throw new Error("unknown run " + RUN); }
header();
const script = document.createElement("script");
script.src = `runs/${RUN}/chat.js`;
script.onload = render;
script.onerror = () => { $("chat").innerHTML = `<p>Conversation file <code>runs/${esc(RUN)}/chat.js</code> is missing from this copy of the report.</p>`; };
document.body.appendChild(script);
