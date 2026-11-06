"""Keyword rules for the agent-strategy taxonomy (multi-label, case-insensitive): model progress notes and bash calls.
Validated against hand labels; see report-taxonomy-validation.json."""
import re
from pathlib import Path

NOTE_RULES = {
    "image": r"silhouette|\brays?\b|\bdepth\b|multi-?view|\bviews\b|(initial|saved|archived|side|close|closer|rotated|reverse|larger) (image|view)|outline|crop|from the images?|color (ratio|mask|region)|regions?\b|back-?project|triangulat|mask|\bscan\b|visible (edge|face|surface)s?|measur\w* .*(shape|dimension|radius|radii|position)",
    "analytic": r"equilibri|torque balance|moment balance|static (load|force|hold|balance)|lever arm|weigh|gravity (load|torque)|force (ratio|balance)|ratio of (the )?(forces|loads)|\bperiod\b|frequenc|natural|logarithmic decrement|decay (rate|ratio)|solv(e|ing) for|\bdecay\b|measur\w* (the )?(grasp )?(force|load|torque|weight|effort)|as expected for|supported weight|supports about|implies|corresponds to (a )?(mass|damping|stiffness)|from the (measured )?(force|load|wrench|torque|effort)s?|(force|load|wrench|torque|effort)s? (give|gives|imply|implies|indicate|yield|determine)|newton|n·m|\bkg\b",
    "fitting": r"\b(fit|fitted|fitting)\b|\brefit|least[- ]squares|optimi[sz]|parameter sweep|\bgrid\b|search|regress|residuals?|objective|minimi[sz]",
    "adjust_check": r"compar|both worlds|in the (candidate|reconstruction)|same (action|grasp|pull|motion|experiment|sequence|release|test|lift|push)|repeat|replay|reproduc|match|agree|validat|\bcheck|held-?out|independent (test|check)|tweak|adjust|try |trying|confirm|verif|\btest(s|ing)?\b|generaliz",
    "prior": r"default|typical|standard|round(ed)? (value|number)|simpler?\b|assum|plausibl|nominal|guess|common (value|choice)|conventional",
}
APPEARANCE = r"light|shadow|colou?r|material|shading|illumin|specular|texture|palette|headlight|rgb|render"

SCRIPT_RULES = {
    "optimizer": r"scipy\.optimize|least_squares|minimize\(|curve_fit|differential_evolution|nelder|golden|brentq|bisect|minimize_scalar|\bOptim|cma\b",
    "sweep": r"for [a-z_, ]+ in (np\.)?(linspace|arange|itertools\.product|product\()|for [a-z_]+ in \[[-0-9.e, ]+\]",
    "simulation": r"mj_step|mujoco\.rollout|mj_forward|MjData\(|import mujoco|mujoco\.Renderer",
    "linear_algebra": r"lstsq|linalg\.solve|polyfit|np\.linalg\.inv|pinv",
    "image": r"PIL|Image\.open|imageio|cv2|\.png",
}


def labels(text, rules):
    return sorted(k for k, r in rules.items() if re.search(r, text, re.I))
SCRIPT_RULES["physics_formula"] = r"9\.8|torque|inertia|\bforces?\b|wrench|\bperiod\b|omega|frequenc|equilibri|lever arm|moment"
SCRIPT_KINDS = ("fit_sim", "fit_data", "simulate", "analytic", "image", "other")
HEREDOC = re.compile(r"cat\s*>\s*(\S+)\s*<<-?\s*['\"]?(\w+)['\"]?\n(.*?)\n\2", re.S)
COPY = re.compile(r"\bcp\s+(\S+\.py)\s+(\S+\.py)")
RUNS = r"(python3?\s+(-\S+\s+)*|exec\(open\(['\"])\S*%s"
IMPORTS = r"(python3?\s+(-\S+\s+)*|exec\(open\(['\"])\S*%s|(from|import)\s+%s\b"


def script_kind(call):
    """One kind per bash call, by priority: optimizer/sweep with local MuJoCo simulation, optimizer/sweep on data or images,
    a plain local simulation, closed-form numerics on physical quantities without image work, image processing, other.
    `call` = (direct, helpers): the command plus scripts it runs, and additionally the helper modules they import (used
    only to detect simulation, since a shared helper module would otherwise lend its optimizer to every call)."""
    direct, helpers = call
    has = set(labels(direct, SCRIPT_RULES))
    sim = "simulation" in has or re.search(SCRIPT_RULES["simulation"], helpers) is not None
    if has & {"optimizer", "sweep"}:
        return "fit_sim" if sim else "fit_data"
    if sim:
        return "simulate"
    if "image" in has:
        return "image"
    if re.search(r"python|import numpy|np\.", direct) and has & {"linear_algebra", "physics_formula"}:
        return "analytic"
    return "other"


def expand(commands):
    """Track script files written by heredoc (and `cp a.py b.py`). For each command return (direct, helpers): the command
    plus the latest content of scripts it executes (`python3 path`, `exec(open(path)`), and the content of helper modules
    those import, up to three levels."""
    files, out = {}, []
    for command in commands:
        for path, _, body in HEREDOC.findall(command):
            files[path] = body
        for a, b in COPY.findall(command):
            if a in files:
                files[b] = files[a]
        ran = lambda text, rule: "".join("\n" + body for path, body in files.items() if body not in text and
                                         re.search(rule % ((re.escape(Path(path).name),) * rule.count("%s") if rule == RUNS else
                                                           (re.escape(Path(path).name), re.escape(Path(path).stem))), text))
        direct = command + ran(command, RUNS)
        helpers = direct
        for _ in range(3):
            helpers += ran(helpers, IMPORTS)
        out.append((direct, helpers))
    return out
