"""Researcher-only reconstruction checkpoint history and standalone PNG curves."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value) if math.isfinite(value) else math.nan


def checkpoint_series(checkpoints):
    """Prepare aligned curves: unavailable values remain NaN, never zero."""
    x = [entry["submission"] for entry in checkpoints]
    chamfer, center = [], []
    primitive_names = []
    density_rows = []
    for entry in checkpoints:
        reconstruction = entry.get("reconstruction") if entry.get("status") == "ok" else None
        reconstruction = reconstruction if isinstance(reconstruction, dict) else {}
        chamfer.append(_number(reconstruction.get("chamfer_distance_m")))
        center.append(_number(reconstruction.get("center_of_mass_error_m")))
        density = reconstruction.get("density", {})
        row = {}
        if isinstance(density, dict):
            for name in density.get("unmatched_source", []):
                if isinstance(name, str) and name not in primitive_names:
                    primitive_names.append(name)
            for primitive in density.get("per_primitive", []):
                name = primitive.get("source_name")
                if not isinstance(name, str):
                    continue
                if name not in primitive_names:
                    primitive_names.append(name)
                row[name] = _number(primitive.get("absolute_error_kg_m3"))
        density_rows.append(row)
    hinge_names=[]
    for entry in checkpoints:
        for name in (entry.get("hinges") or {}):
            if name not in hinge_names:
                hinge_names.append(name)
    hinges={}
    for name in hinge_names:
        hinges[name]={}
        for parameter in ("damping", "frictionloss", "stiffness"):
            values=[]
            for entry in checkpoints:
                hinge=(entry.get("hinges") or {}).get(name,{})
                values.append(_number(hinge.get(parameter+"_absolute_error"))
                    if entry.get("status")=="ok" and hinge.get("status")=="ok" else math.nan)
            hinges[name][parameter+"_absolute_error"]=values
    return {"submission": x, "hinges": hinges, "chamfer_distance_m": chamfer,
            "center_of_mass_error_m": center,
            "density_absolute_error_kg_m3": {name: [row.get(name, math.nan) for row in density_rows]
                                               for name in primitive_names}}


def _plot(history, destination):
    # Direct Agg canvases avoid global pyplot state and work without a display.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    from matplotlib import colormaps

    entries = history["checkpoints"]
    series = checkpoint_series(entries)
    figure = Figure(figsize=(12, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    x = series["submission"]
    count = len(x)
    scalar_panels = ((axes[0], "chamfer_distance_m", "Chamfer distance", "Distance (m)"),
                     (axes[2], "center_of_mass_error_m", "Center of mass error", "Distance (m)"))
    summary = {}
    for axis, key, title, ylabel in scalar_panels:
        values = series[key]
        axis.plot(x, values, marker="o", linewidth=1.7)
        missing = sum(not math.isfinite(value) for value in values)
        summary[key] = {"unavailable_checkpoints": missing, "checkpoints": count}
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.text(.99, .95, f"{missing}/{count} checkpoints unavailable", transform=axis.transAxes,
                  ha="right", va="top", fontsize=9, color="#555555")
    density = series["density_absolute_error_kg_m3"]
    density_axis = axes[1]
    for index, (name, values) in enumerate(density.items()):
        density_axis.plot(x, values, marker="o", linewidth=1.4, label=name,
                          color=colormaps["tab20"](index % 20))
    missing_pairs = sum(not math.isfinite(value) for values in density.values() for value in values)
    unavailable_checkpoints = sum(not any(math.isfinite(values[i]) for values in density.values()) for i in range(count))
    summary["density"] = {"unavailable_checkpoints": unavailable_checkpoints,
                          "unavailable_primitive_checkpoint_pairs": missing_pairs,
                          "primitive_count": len(density), "checkpoints": count}
    density_axis.set_title("Density absolute error by original primitive", loc="left")
    density_axis.set_ylabel("Density error (kg/m³)")
    density_axis.text(.99, .95, f"{unavailable_checkpoints}/{count} checkpoints unavailable; {missing_pairs} primitive gaps",
                      transform=density_axis.transAxes, ha="right", va="top", fontsize=9, color="#555555")
    if density:
        density_axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=.25)
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if count == 1:
            axis.set_xlim(x[0] - .5, x[0] + .5)
    axes[-1].set_xlabel("Explicit reconstruction checkpoint / version")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(entry["submission"]) + "\n" + (str(entry["version"])[:10] if entry.get("version") else "no version")
                             for entry in entries], fontsize=8)
    figure.suptitle("Private reconstruction history — gaps indicate unavailable metrics", fontsize=13)
    temporary = destination.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png", dpi=150)
    temporary.replace(destination)
    figure.clear()
    return summary


def _plot_hinges(history, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    series=checkpoint_series(history['checkpoints'])
    if not series['hinges']:
        return None
    figure=Figure(figsize=(10,10),constrained_layout=True)
    FigureCanvasAgg(figure)
    axes=figure.subplots(3,1,sharex=True)
    summary={}
    for axis,parameter,label in zip(axes,('damping','frictionloss','stiffness'),('Damping error (N·m·s/rad)','Friction loss error (N·m)','Stiffness error (N·m/rad)')):
        missing=0
        for name,values in series['hinges'].items():
            y=values[parameter+'_absolute_error']
            axis.plot(series['submission'],y,marker='o',label=name)
            missing+=sum(not math.isfinite(value) for value in y)
        axis.set_ylabel(label)
        axis.set_title('Hinge '+parameter+' absolute error',loc='left')
        axis.set_ylim(bottom=0);axis.grid(True,alpha=.25)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(frameon=False)
        axis.text(.99,.95,f'{missing} unavailable hinge/checkpoint pairs',transform=axis.transAxes,ha='right',va='top',fontsize=8)
        summary[parameter]={'unavailable_hinge_checkpoint_pairs':missing}
    axes[-1].set_xlabel('Explicit reconstruction checkpoint / version')
    axes[-1].set_xticks(series['submission'])
    axes[-1].set_xticklabels([str(e['submission'])+'\n'+str(e.get('version') or 'no version')[:10] for e in history['checkpoints']],fontsize=8)
    figure.suptitle('Private hinge parameter history — gaps indicate unavailable metrics')
    temporary=destination.with_suffix('.png.tmp')
    figure.savefig(temporary,format='png',dpi=150);temporary.replace(destination);figure.clear()
    return summary


def update_metric_history(output_dir, submission, report, *, render_plots=True):
    """Append or replace a checkpoint and refresh researcher-only JSON/PNG artifacts.

    The complete reconstruction object is retained, including both COM vectors,
    matched densities, unmatched source names and component error/status details.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "metric-history.json"
    history = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "checkpoints": []}
    scores = report.get("scores")
    reconstruction = scores.get("reconstruction") if isinstance(scores, dict) else None
    hinges = scores.get("hinges") if isinstance(scores, dict) else None
    entry = {"submission": int(submission), "version": report.get("version"),
             "status": report.get("status", "failed"), "reconstruction": reconstruction, "hinges": hinges,
             "reset": scores.get("reset") if isinstance(scores, dict) else None,
             "unavailable_reason": report.get("error") or report.get("errors") or
                 ("Reconstruction metrics unavailable or disabled" if reconstruction is None else None)}
    history["checkpoints"] = [old for old in history["checkpoints"] if old["submission"] != int(submission)] + [entry]
    history["checkpoints"].sort(key=lambda old: old["submission"])
    # Save raw measurements even if an installation/plotting error occurs later.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2, allow_nan=False))
    temporary.replace(path)
    if render_plots:
        history["plot_summary"] = _plot(history, output / "metric-history.png")
        history["hinge_plot_summary"] = _plot_hinges(history, output / "hinge-history.png")
    temporary.write_text(json.dumps(history, indent=2, allow_nan=False))
    temporary.replace(path)
    return history


def plot_com_trajectories(destination, name, sample_seconds, source, candidate):
    """Researcher-only PNG of one arm probe's composite object COM: x, y, z over time and the source-candidate distance.
    A model without free object mass has None samples, drawn as gaps."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = lambda samples: np.array([[math.nan] * 3 if p is None else p for p in samples], dtype=float).reshape(-1, 3)
    a, b = rows(source), rows(candidate)
    t = np.arange(len(a)) * sample_seconds
    figure = Figure(figsize=(9, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 1, sharex=True)
    for k, axis in enumerate(axes[:3]):
        axis.plot(t, a[:, k], color="#2a78d6", linewidth=2, label="source")
        axis.plot(t, b[:, k], color="#eb6834", linewidth=2, linestyle="--", label="candidate")
        axis.set_ylabel(f"COM {'xyz'[k]} (m)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[3].plot(t, np.linalg.norm(a - b, axis=1), color="#2a78d6", linewidth=2)
    axes[3].set_ylabel("distance (m)")
    axes[3].set_ylim(bottom=0)
    axes[3].set_xlabel("probe time (s)")
    for axis in axes:
        axis.grid(True, alpha=.25)
    figure.suptitle(f"Composite object centre of mass — arm probe {name}", fontsize=12)
    figure.savefig(destination, format="png", dpi=110)
    figure.clear()
