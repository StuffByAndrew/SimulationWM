"""Researcher results report builder (reusable template), organized by scientific question. Three steps:
  build-results-report.py data SPEC.py WORK_DIR OUT_DIR   slow (minutes): per-run numbers, 3D models, run pages, videos (rerun when runs change)
  build-results-report.py text SPEC.py OUT_DIR            seconds: all prose, captions, sections, run names and page code (rerun for any edit)
  build-results-report.py zip OUT_DIR                     OUT_DIR.zip for sharing
SPEC (e.g. report-spec-20261104.py) names the runs, input files, comparisons and sections; this file is generic.
Inputs are researcher-side JSON made on a compute node: models (report-extract-models.py: MuJoCo-compiled source and
submissions), checkpoint indexes (checkpoint-index.py: wall minutes and cumulative tokens per scored checkpoint) and traces
(report-scan-audit.py: model replies, tool calls, controller actions). This script reads JSON only, writes OUT_DIR
(index.html, report.js, data.js, vendor/, videos/) plus OUT_DIR.zip, and never parses MJCF or runs simulation.
Private: never mount the output in an agent sandbox."""
import importlib.util, json, math, os, re, shutil, sys
from collections import Counter
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parent
sys.path.insert(0, str(VAL))
TAX = __import__("report-taxonomy-rules")
at = __import__("checkpoint-index").at
ROOTS = (VAL.parent / "runs", Path("/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"),
         Path("/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments/runs"))  # home, lab, replicates
POINTS = ("final", "at_120min", "at_matched_tokens")
# Metric keys of one submission. "ratio": compiled estimate ÷ true (1 = exact); "err": lower is better.
METRICS = {"chamfer_mm": ("Chamfer distance (mm)", "err"), "com_mm": ("COM error (mm)", "err"), "rgb_mse": ("image MSE, hidden view", "err"),
           "damping_rel": ("hinge damping error (scored)", "err"), "frictionloss_rel": ("hinge dry-friction error (scored)", "err"),
           "sphere_err": ("worst per-sphere mass error", "err"), "r:mass": ("mass", "ratio"), "r:inertia": ("inertia about hinge", "ratio"),
           "r:damping": ("hinge damping", "ratio"), "r:frictionloss": ("hinge dry friction", "ratio"), "r:stiffness": ("hinge stiffness", "ratio"),
           "r:massratio": ("mass ratio between the two blocks", "ratio")}


def run_dir(run):
    return next(p / run for p in ROOTS if (p / run / "metric-history.json").is_file())


def merged(paths):
    out = {}
    for p in paths:
        for k, v in json.loads(Path(p).read_text()).items():
            assert out.get(k, v) == v, (p, k)  # the same run extracted twice must agree
            out[k] = v
    return out


def match(src, cand, key):
    """Greedy nearest match of source items to candidate items by world position."""
    out, used = {}, set()
    for dist, i, j in sorted((math.dist(s[key], c[key]), i, j) for i, s in enumerate(src) for j, c in enumerate(cand)):
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def estimates(source, m):
    """Physical estimates of compiled model m aligned to the source. Hinge scenes: the hinge body's mass, its inertia about the
    axis, and every joint parameter that is nonzero in the source. Free bodies: mass of each source body's nearest candidate
    body (by COM), the two-block mass ratio, and for a one-body cluster of four spheres the sphere masses implied by total
    mass and COM at the candidate's own sphere centers (four non-coplanar centers determine them exactly)."""
    out = {}
    if source["joints"]:
        assert len(source["joints"]) == len(m["joints"]) == 1, [j["name"] for j in m["joints"]]
        js, jm = source["joints"][0], m["joints"][0]
        out.update(mass=next(b["mass"] for b in m["bodies"] if b["name"] == jm["body"]), inertia=jm["axis_inertia"])
        out.update({k: jm[k] for k in ("damping", "frictionloss", "stiffness") if js[k] > 0})
        return out
    bodies, pairing = source["bodies"], match(source["bodies"], m["bodies"], "xipos")
    masses = [m["bodies"][pairing[i]]["mass"] if i in pairing else None for i in range(len(bodies))]
    if len(bodies) == 1:
        out["mass"] = masses[0]
    else:
        out.update({f"mass:{b['name']}": x for b, x in zip(bodies, masses)})
    if len(bodies) == 2 and None not in masses:
        out["massratio"] = masses[0] / masses[1]
    spheres = [g for g in source["geoms"] if g["type"] == "sphere"]
    if len(bodies) == 1 and len(spheres) == 4:
        cand = [g for g in m["geoms"] if g["type"] == "sphere" and g["rgba"][3] > 0]
        p = match(spheres, cand, "pos")
        A = np.vstack([np.ones(4), np.array([cand[p[i]]["pos"] for i in range(4)]).T])
        b = np.r_[m["total_mass"], m["total_mass"] * np.asarray(m["com"])]
        out.update({f"sphere:{spheres[i]['name']}": float(x) for i, x in enumerate(np.linalg.solve(A, b))})
    return out


def volume(g):
    s = g["size"]
    return {"sphere": lambda: 4 / 3 * math.pi * s[0] ** 3, "ellipsoid": lambda: 4 / 3 * math.pi * s[0] * s[1] * s[2], "box": lambda: 8 * s[0] * s[1] * s[2],
            "cylinder": lambda: 2 * math.pi * s[0] ** 2 * s[1], "capsule": lambda: 2 * math.pi * s[0] ** 2 * s[1] + 4 / 3 * math.pi * s[0] ** 3}.get(g["type"], lambda: None)()


def mass_parts(source, m):
    """Where a model's object mass sits: the bodies estimates() reads (hinge body, or each source body's nearest candidate body)
    and their visible geoms with own mass (None under an explicit body inertial) and density."""
    names = ({j["body"] for j in m["joints"][:1]} if source["joints"] else  # a model without a joint has no hinge body
             {m["bodies"][j]["name"] for j in match(source["bodies"], m["bodies"], "xipos").values()})
    return dict(bodies={b["name"]: b["mass"] for b in m["bodies"] if b["name"] in names},
                geoms=[dict(name=g["name"] or g["type"], body=g["body"], mass=g["mass"], density=g["mass"] / volume(g) if g["mass"] is not None and volume(g) else None)
                       for g in m["geoms"] if g["rgba"][3] > 0 and g["body"] in names])


def submission(source, model, hist, truth, est_fn=estimates):
    if hist["status"] == "failed":  # rejected by the evaluator (e.g. a modified robot fixture): a scored gap, with the evaluator's reason
        return dict(m={}, est={}, rejected=hist["unavailable_reason"])
    assert "missing" not in model, "a submission without its entry scene must be one the evaluator rejected"
    assert hist["status"] == "ok", hist["status"]
    rec, hinge = hist.get("reconstruction") or {}, next(iter((hist.get("hinges") or {}).values()), None) or {}
    est = est_fn(source, model)
    m = dict(chamfer_mm=None if rec.get("chamfer_distance_m") is None else 1000 * rec["chamfer_distance_m"],
             rgb_mse=(hist.get("reset") or {}).get("rgb_mse"), damping_rel=hinge.get("damping_relative_error"),
             frictionloss_rel=hinge.get("frictionloss_relative_error"), com_mm=None)
    if source["joints"]:  # jointed scenes: the evaluator's object COM error where it reports one
        m["com_mm"] = None if rec.get("center_of_mass_error_m") is None else 1000 * rec["center_of_mass_error_m"]
    elif model["com"] is None:  # no object body (an arm-only draft): a gap, as the evaluator reports it
        assert not model["bodies"] and rec.get("center_of_mass_status") == "unavailable", hist["version"]
    else:  # compiled COM of the object bodies; must equal the evaluator's wherever it reports one
        m["com_mm"] = 1000 * math.dist(model["com"], source["com"])
        assert rec.get("center_of_mass_error_m") is None or abs(rec["center_of_mass_error_m"] * 1000 - m["com_mm"]) < 0.01
    # estimate ÷ true; a parameter that is zero in the truth (e.g. an undamped spring's damping) keeps its absolute value as abs:<name>
    m.update({(f"r:{k}" if truth[k] else f"abs:{k}"): None if v is None else v / truth[k] if truth[k] else v for k, v in est.items()})
    spheres = [k for k in est if k.startswith("sphere:")]
    if spheres:
        m["sphere_err"] = max(abs(est[k] / truth[k] - 1) for k in spheres)
    return dict(m=m, est=est)


def trace_stats(trace, tools):
    """Controller actions per world, tool calls by name, approximate wall time each call kind occupies (time from the reply
    that issued it to the next reply), every bash call (with the scripts it runs, for classification) and agent-tool use."""
    responses = [e for e in trace if e["kind"] == "response"]
    actions = Counter((e["op"], e["world"]) for e in trace if e["kind"] == "action" and e["op"] not in ("inference", "progress"))
    calls, seconds, scripts = Counter(), Counter(), []
    for a, b in zip(responses, responses[1:] + [None]):
        names = [t["name"] for t in (a["tool_calls"] or [])]
        for t in a["tool_calls"] or []:
            calls[t["name"]] += 1
            if t["name"] == "bash":
                scripts.append(dict(t=a["t"], command=json.loads(t["arguments"])["command"]))
        if b is not None:
            for n in names:
                seconds[n] += (b["t"] - a["t"]) / len(names)
    for s, call in zip(scripts, TAX.expand([s["command"] for s in scripts])):
        s["call"] = call
    uptake = {}
    for name, rule in tools.items():
        hits = [s for s in scripts if re.search(rule, s["call"][0])]
        uptake[name] = dict(calls=len(hits), help=sum(bool(re.search(r"--help", s["command"])) for s in hits),
                            first_min=hits[0]["t"] / 60 if hits else None)
    return dict(actions={f"{op}|{w}": n for (op, w), n in actions.items()}, calls=dict(calls), seconds=dict(seconds), scripts=scripts,
                replies=len(responses), tools=uptake)


def taxonomy(notes, scripts):
    note_labels = [TAX.labels(n["text"], TAX.NOTE_RULES) for n in notes]
    script_kinds = [TAX.script_kind(s["call"]) for s in scripts]
    return dict(notes=len(notes), note_counts=dict(Counter(l for ls in note_labels for l in ls)),
                appearance=sum(bool(re.search(TAX.APPEARANCE, n["text"], re.I)) for n in notes),
                scripts=len(scripts), script_counts=dict(Counter(script_kinds))), note_labels, script_kinds


class Ctx:
    """What a spec's text and section functions see: the per-run data plus small formatting helpers."""
    def __init__(self, runs, comparisons):
        self.runs, self.comparisons = runs, comparisons

    def final(self, run, key):
        return self.runs[run]["subs"][-1]["m"][key]

    def first(self, run, key):
        return self.runs[run]["subs"][0]["m"][key]

    @staticmethod
    def x(r, d=3):
        return "0 (dropped)" if r == 0 else "exact" if abs(r - 1) < 1e-3 else f"×{r:.{d}g}"

    @staticmethod
    def mm(v, d=3):
        return "≈0 mm" if v < 0.005 else f"{v:.{d}g} mm"


def load_spec(spec_path):
    spec_mod = importlib.util.spec_from_file_location("spec", spec_path)
    S = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(S)
    return S


PAGE_FILES = (("results-report.html", "index.html"), ("results-report.js", "report.js"), ("results-report.css", "style.css"),
              ("results-report-common.js", "common.js"), ("results-run.html", "run.html"), ("results-run.js", "run.js"))


def text(spec_path, out):
    """Seconds: everything the spec writes as prose (meta, glossary, sections, captions, strategy, methods, run names, comparison
    labels) into OUT/text.js, plus the page code and saved edits. Reads the per-run numbers from OUT/data.js (written by `data`)."""
    S, out = load_spec(spec_path), Path(out)
    R = json.loads((out / "data.js").read_text()[len("window.R="):-1])
    assert set(R["runs"]) == set(S.NAMES), "runs changed: rerun `data`"
    for cid, c in S.COMPARISONS.items():
        assert R["comparisons"][cid]["base"] == c["base"] and R["comparisons"][cid]["abl"] == c["abl"], f"comparison {cid} changed: rerun `data`"
    for run, (scene, condition) in S.NAMES.items():  # names are prose: the spec's current ones
        R["runs"][run].update(scene=scene, condition=condition)
    for cid, c in S.COMPARISONS.items():
        R["comparisons"][cid].update(c)
    ctx = Ctx(R["runs"], R["comparisons"])
    T = dict(meta=S.META, glossary=S.GLOSSARY, sections=S.sections(ctx), strategy=S.strategy(ctx), methods=S.methods(ctx),
             names=S.NAMES, comparisons={cid: {k: c[k] for k in ("label", "base_label", "abl_label", "keys")} for cid, c in S.COMPARISONS.items()},
             example_names={ex["text"]: " · ".join(S.NAMES[ex["run"]]) for items in S.EXAMPLES.values() for ex in items})
    for src, dst in PAGE_FILES:
        shutil.copy(VAL / src, out / dst)
    edits = json.loads(Path(S.EDITS).read_text())  # text edits saved from the report's edit mode (serve-report.py)
    assert isinstance(edits, dict) and all(isinstance(v, str) for v in edits.values())
    (out / "edits.js").write_text("window.EDITS=" + json.dumps(edits, ensure_ascii=False) + ";")
    (out / "text.js").write_text("window.T=" + json.dumps(T, separators=(",", ":"), ensure_ascii=False) + ";")
    print(T["strategy"]["answer"])


def zip_report(out):
    """OUT.zip for sharing (slow: the run pages hold about 450 MB of images)."""
    out = Path(out)
    assert (out / "data.js").is_file() and (out / "text.js").is_file(), out
    shutil.make_archive(str(out), "zip", out.parent, out.name)


def data(spec_path, work, out):
    """Slow: per-run numbers, comparisons, 3D models and taxonomy into OUT/data.js, plus videos, vendor code and run pages. OUT must not exist."""
    S = load_spec(spec_path)
    work, out = Path(work), Path(out)
    assert not out.exists(), out
    # a SUMMARIES entry is a file (every completed run in it is reported) or (file, runs): only those runs, e.g. baselines from an earlier report
    entries = [(f, None) if isinstance(f, (str, Path)) else f for f in S.SUMMARIES]
    rows = {r["run"]: r for f, only in entries for r in json.loads((VAL / f).read_text())
            if r["status"] == "completed" and r["run"] not in S.INVALID and (only is None or r["run"] in only)}
    assert set(rows) == set(S.NAMES), set(rows) ^ set(S.NAMES)
    excluded = json.loads((VAL / "excluded-runs.json").read_text())["runs"]
    for run in S.NAMES:  # results use only strict (sandboxed, API) runs; development-mode runs are excluded by user decision
        assert run not in excluded, run
        assert json.loads((run_dir(run) / "experiment.json").read_text())["execution_mode"] == "strict_oci", run
    models, cps = merged(S.MODELS), merged(S.CHECKPOINTS)
    runs, labels_out = {}, {}
    for run in S.NAMES:
        model = models[run]
        trace = json.loads(next(Path(d) / f"{run}.json" for d in S.TRACES if (Path(d) / f"{run}.json").is_file()).read_text())
        notes = [dict(t=x["wall_elapsed"], utc=x["utc_time"], text=x["text"]) for x in map(json.loads, (run_dir(run) / "progress.jsonl").read_text().splitlines())
                 if x.get("origin") == "model"]
        stats = trace_stats(trace, S.TOOLS)
        hist = json.loads((run_dir(run) / "metric-history.json").read_text())["checkpoints"]
        cp = cps[run]["checkpoints"]
        assert len(hist) == len(cp) == len(model["submissions"]) == rows[run]["submissions"], run
        assert [c["submission"] for c in cp] == [h["submission"] for h in hist], run
        offset = rows[S.CONTINUATIONS[run]]["wall_min"] if run in S.CONTINUATIONS else 0.0
        est_fn = (lambda src, m: S.estimates(run, src, m, estimates)) if hasattr(S, "estimates") else estimates  # optional spec hook: scene-specific estimates
        truth = est_fn(model["source"], model["source"])
        subs = [dict(t=c["wall_min"] + offset, mtok=c["tokens"] / 1e6, **submission(model["source"], s, h, truth, est_fn)) for c, s, h in zip(cp, model["submissions"], hist)]
        for i, sub in enumerate(subs):  # optional spec hook: extra per-submission metrics (keys declared in S.METRICS)
            sub["m"].update(S.extra_metrics(run, i, hist[i]) if hasattr(S, "extra_metrics") else {})
        tax, note_labels, script_kinds = taxonomy(notes, stats["scripts"])
        labels_out[run] = dict(notes=[dict(t=n["t"], utc=n["utc"], labels=l) for n, l in zip(notes, note_labels)],
                               scripts=[dict(t=s["t"], kind=k) for s, k in zip(stats["scripts"], script_kinds)])
        flags = {k: [dict(min=round(n["t"] / 60), text=n["text"]) for n in notes if re.search(rule, n["text"], re.I)] for k, rule in S.NOTE_FLAGS.items()}
        row = rows[run]
        runs[run] = dict(scene=S.NAMES[run][0], condition=S.NAMES[run][1], factor=row["factor"], wall_min=row["wall_min"], budget_used=row["budget_used"],
                         cost_usd=row["cost_usd"], reason=row["reason"], budget_stop=row["budget_stop"], submissions=row["submissions"],
                         final_mtok=cps[run]["final_tokens"] / 1e6, truth=truth, subs=subs, actions=stats["actions"], calls=stats["calls"],
                         call_seconds=stats["seconds"], taxonomy=tax, parts=dict(source=mass_parts(model["source"], model["source"]),
                                                                           final=None if "missing" in model["submissions"][-1] else mass_parts(model["source"], model["submissions"][-1])), tools=stats["tools"], flags=flags, replies=stats["replies"])
    comparisons = {}
    for cid, c in S.COMPARISONS.items():
        members = c["base"] + [c["abl"]]
        tokens = min(cps[r]["final_tokens"] for r in members)
        pts = {}
        for point, kw in zip(POINTS, (dict(wall_min=1e9), dict(wall_min=120), dict(tokens=tokens))):
            pts[point] = {}
            for r in members:
                hit = at(cps[r]["checkpoints"], **kw)
                pts[point][r] = None if hit is None else cps[r]["checkpoints"].index(hit)
        comparisons[cid] = dict(c, mtok=tokens / 1e6, points=pts)
    for cat, items in S.EXAMPLES.items():
        for ex in items:
            hit = [x for x in map(json.loads, (run_dir(ex["run"]) / "progress.jsonl").read_text().splitlines()) if x.get("origin") == "model" and x["text"] == ex["text"]]
            assert len(hit) == 1, ex
            ex.update(utc=hit[0]["utc_time"], minute=round(hit[0]["wall_elapsed"] / 60), name=" · ".join(S.NAMES[ex["run"]]))
    viewer = {}
    geoms = lambda m: [{k: g[k] for k in ("type", "size", "pos", "quat", "rgba", "mesh") if k in g} for g in m["geoms"] if g["rgba"][3] > 0]
    for group, members in S.VIEWER.items():
        for run in members:
            chain = [run] + [c for c, parent in S.CONTINUATIONS.items() if parent == run]
            viewer[run] = dict(group=group, source=dict(geoms=geoms(models[run]["source"]), com=models[run]["source"]["com"]),
                               subs=[dict(label=("+2 h · " if r != run else "") + f"submission {i + 1}" + (" (rejected)" if "rejected" in runs[r]["subs"][i] else ""),
                                          geoms=geoms(s), com=s["com"], t=runs[r]["subs"][i]["t"],
                                          chamfer=runs[r]["subs"][i]["m"].get("chamfer_mm"), com_err=runs[r]["subs"][i]["m"].get("com_mm"))
                                     for r in chain for i, s in enumerate(models[r]["submissions"])])
    validation = json.loads((VAL / "report-taxonomy-validation.json").read_text())
    videos = json.loads((Path(S.VIDEOS) / "data.json").read_text())["videos"] if hasattr(S, "VIDEOS") else []
    R = dict(metrics={**METRICS, **getattr(S, "METRICS", {})}, runs=runs, order=S.ORDER, continuations=S.CONTINUATIONS, comparisons=comparisons,
             viewer=viewer, examples=S.EXAMPLES, taxonomy_validation=validation, videos=[v for v in videos if v["run"] in runs])
    (out / "vendor").mkdir(parents=True)
    (out / "videos").mkdir()
    if hasattr(S, "VIDEOS"):
        shutil.copytree(Path(S.VIDEOS) / "videos", out / "videos", dirs_exist_ok=True)
    for name, src in getattr(S, "EXTRA_VIDEOS", {}).items():  # optional spec field: {published name: recorded mp4}
        shutil.copy(src, out / "videos" / name)
    if hasattr(S, "FIGURES"):  # optional spec field: {published name: researcher PNG}, served as figs/<name>
        (out / "figs").mkdir()
        for name, src in S.FIGURES.items():
            (out / "figs" / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, out / "figs" / name)
    for f in ("three.min.js", "OrbitControls.js"):
        shutil.copy(Path(S.VENDOR) / f, out / "vendor" / f)
    chats = (S.CHAT,) if isinstance(S.CHAT, (str, Path)) else S.CHAT  # one directory or several
    for run in S.NAMES:  # per-run pages: the full conversation and the images the model received (report-chat-extract.py)
        src = next(Path(d) / run for d in chats if (Path(d) / run / "chat.js").is_file())
        same_fs = os.stat(src).st_dev == os.stat(out).st_dev  # hard links need one filesystem (some chat pages live on lab storage)
        shutil.copytree(src, out / "runs" / run, copy_function=os.link if same_fs else shutil.copy2)
    (out / "data.js").write_text("window.R=" + json.dumps(R, separators=(",", ":")) + ";")
    (out / "taxonomy-labels.json").write_text(json.dumps(dict(rules=dict(notes=TAX.NOTE_RULES, appearance=TAX.APPEARANCE, scripts=TAX.SCRIPT_RULES),
                                                             validation=validation, labels=labels_out), indent=1))


if __name__ == "__main__":
    step, args = sys.argv[1], sys.argv[2:]
    {"data": data, "text": text, "zip": zip_report}[step](*args)
