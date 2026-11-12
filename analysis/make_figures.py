"""Grafieken en samenvattende tabel voor het verslag (deelvragen 5 tot 8).

Leest de meetresultaten die er al zijn en slaat over wat nog ontbreekt:
  results/offline_jetson/offline_metrics.csv   offline evaluatie op de Jetson (deelvraag 5)
  results/chain/chain_<model>_<prec>_*.csv     ketentest, 500 frames per engine (deelvraag 6)
  results/live/live_<model>_<prec>_*.csv       live landingen, 50 per engine (deelvragen 7 en 8)
Per engine telt telkens het nieuwste bestand. De mappen in results/live/ (oude, mislukte
runs) worden niet gelezen.

  results/robustness_pc/robustness_metrics.csv  moeilijkere testsets, PyTorch op de pc (stap 1)
  results/robustness_jetson/robustness_metrics.csv  moeilijkere testsets, 6 engines op de Jetson
  results/threshold_sweep/sweep.csv           recall en vals alarm per drempel (analysis/threshold_sweep.py)
Resultaat: PNG's in results/figures/ en results/summary.csv (één rij per engine).

Gebruik (vanuit de projectmap; train-env heeft matplotlib, airsim-env niet):
    train-env\\Scripts\\python.exe analysis\\make_figures.py
    train-env\\Scripts\\python.exe analysis\\make_figures.py --dark   # donkere versie voor presentation/
De donkere versie komt in results/figures_dark/ (zelfde data, andere kleuren van tekst en achtergrond).
"""
import csv
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # enkel bestanden schrijven, geen venster
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_DIR / "results"
FIGURES = RESULTS / "figures"

MODELS = ["yolo26s", "rtdetr-l"]
MODEL_LABELS = {"yolo26s": "YOLO26s", "rtdetr-l": "RT-DETR-l"}
PRECISIONS = ["fp32", "fp16", "int8"]
FPS_REQUIREMENT = 20

# Eén kleur per rekenprecisie, in elke grafiek dezelfde (gecontroleerd op kleurenblindheid).
# Het model staat telkens in de titel van het paneel of op de as, nooit enkel in de kleur.
PRECISION_COLORS = {"fp32": "#2a78d6", "fp16": "#eb6834", "int8": "#1baf7a"}
PRECISION_MARKERS = {"fp32": "o", "fp16": "s", "int8": "^"}
TEXT = "#0b0b0b"
MUTED = "#898781"
GRID = "#e4e3df"
BG = "white"            # achtergrond; ook de witte rand tussen staven
INK_ON_LIGHT = "#0b0b0b"   # tekst in lichte vakjes van een heatmap, in beide thema's

# Uitkomsten: statuskleuren (goed / twijfel / fout) met de naam erbij in de legende.
OUTCOMES = ["goede_landing", "terecht_afgebroken", "onterecht_afgebroken", "slechte_landing", "foute_landing"]
OUTCOME_COLORS = {"goede_landing": "#0ca30c", "terecht_afgebroken": "#7fc97f",
                  "onterecht_afgebroken": "#fab219", "slechte_landing": "#ec835a",
                  "foute_landing": "#d03b3b"}
CORRECT = {"goede_landing", "terecht_afgebroken"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest(pattern):
    """Nieuwste bestand dat past (de tijdstempel staat in de naam, dus sorteren volstaat)."""
    paths = sorted(RESULTS.glob(pattern))
    return paths[-1] if paths else None


def engines():
    return [(m, p) for m in MODELS for p in PRECISIONS]


def label(model, precision):
    return f"{MODEL_LABELS[model]} {precision.upper()}"


# ---------------------------------------------------------------- offline (deelvraag 5)

def load_offline():
    path = RESULTS / "offline_jetson" / "offline_metrics.csv"
    return read_csv(path) if path.exists() else []


def offline_value(rows, model, precision, group, value, metric):
    for r in rows:
        if r["model"] == model and r["precision_mode"] == precision and r["group"] == group \
                and (value is None or float(r["value"]) == value):
            return float(r[metric])
    return np.nan


def plot_offline_per_group(rows):
    """mAP50-95 per afstand en per hoek. mAP50 staat overal op ± 0,995 (plafond), dus
    het verschil tussen de rekenprecisies zit enkel in hoe nauwkeurig de box zit."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for col, model in enumerate(MODELS):
        for row, (group, values, xlabel) in enumerate([
                ("distance", [5, 10, 20, 30], "Afstand d (m)"),
                ("theta", [0, 15, 30, 45], "Naderingshoek θ (°)")]):
            ax = axes[row][col]
            for p in PRECISIONS:
                y = [offline_value(rows, model, p, group, v, "map50_95") for v in values]
                # FP32 groter getekend: bij YOLO26s vallen FP32 en FP16 exact samen.
                ax.plot(values, y, color=PRECISION_COLORS[p], marker=PRECISION_MARKERS[p],
                        linewidth=2, markersize=10 if p == "fp32" else 6, label=p.upper())
            ax.set_xticks(values)
            ax.set_xlabel(xlabel)
            ax.set_title(MODEL_LABELS[model], loc="left", fontweight="bold")
            if col == 0:
                ax.set_ylabel("mAP50-95 (testset)")
    axes[0][0].set_ylim(0.80, 1.0)
    axes[0][0].legend(loc="lower left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    labels[0] = "FP32 (valt samen met FP16)"
    axes[0][0].legend(handles, labels, loc="lower left")
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle("Offline op de Jetson: mAP50-95 per afstand en per hoek", x=0.06, ha="left", fontweight="bold")
    fig.savefig(FIGURES / "offline_map50_95.png")
    plt.close(fig)


def plot_offline_heatmaps(rows):
    """mAP50-95 per combinatie θ × d, één paneel per engine (zelfde kleurschaal)."""
    thetas, distances = [0, 15, 30, 45], [5, 10, 20, 30]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    for (model, p), ax in zip(engines(), axes.flat):
        grid = np.full((len(thetas), len(distances)), np.nan)
        for r in rows:
            if r["model"] == model and r["precision_mode"] == p and r["group"] == "theta_x_distance":
                t, d = (int(v) for v in r["value"].split("_"))
                grid[thetas.index(t), distances.index(d)] = float(r["map50_95"])
        im = ax.imshow(grid, cmap="Blues", vmin=0.80, vmax=1.0, origin="lower")
        for i in range(len(thetas)):
            for j in range(len(distances)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.93 else INK_ON_LIGHT)
        ax.set_xticks(range(len(distances)), distances)
        ax.set_yticks(range(len(thetas)), thetas)
        ax.grid(False)
        ax.set_title(label(model, p), loc="left", fontweight="bold")
    for ax in axes[1]:
        ax.set_xlabel("Afstand d (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Hoek θ (°)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mAP50-95")
    fig.suptitle("Offline: mAP50-95 per hoek en afstand", x=0.06, ha="left", fontweight="bold")
    fig.savefig(FIGURES / "offline_map50_95_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------- keten (deelvraag 6)

def load_chain():
    chain = {}
    for model, p in engines():
        path = latest(f"chain/chain_{model}_{p}_*.csv")
        if path:
            rows = read_csv(path)
            chain[(model, p)] = {k: np.array([float(r[k]) for r in rows])
                                 for k in ["frame_ms", "capture_ms", "inference_ms", "server_ms", "network_ms"]}
    return chain


def fps_stats(frame_ms):
    """Gemiddelde FPS = 1000 / gemiddelde frametijd; 5e percentiel = de 5 % traagste frames."""
    return 1000 / frame_ms.mean(), 1000 / np.percentile(frame_ms, 95)


def grouped_bars(ax, values, fmt):
    """Staven per model (groep) en per rekenprecisie (kleur), met de waarde erboven."""
    width = 0.26
    for i, p in enumerate(PRECISIONS):
        x = np.arange(len(MODELS)) + (i - 1) * (width + 0.02)
        y = [values.get((m, p), np.nan) for m in MODELS]
        bars = ax.bar(x, y, width, color=PRECISION_COLORS[p], label=p.upper())
        for b, v in zip(bars, y):
            if not np.isnan(v):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    ax.grid(axis="x", visible=False)


def plot_chain_fps(chain, live_fps):
    """Keten-FPS (ketentest) per engine, met de eis van 20 FPS. Live-FPS ernaast als controle."""
    fig, axes = plt.subplots(1, 2 if live_fps else 1, figsize=(10 if live_fps else 6, 4.2),
                             sharey=True, squeeze=False)
    panels = [("Ketentest (500 frames, hangend op 10 m)", {k: fps_stats(v["frame_ms"])[0] for k, v in chain.items()})]
    if live_fps:
        panels.append(("Tijdens de live landingen", live_fps))
    for ax, (title, values) in zip(axes[0], panels):
        grouped_bars(ax, values, "{:.1f}")
        ax.axhline(FPS_REQUIREMENT, color=TEXT, linewidth=1, linestyle="--")
        # Rechts boven de lijn: daar staan de lagere RT-DETR-staven, dus geen botsing met waarden.
        ax.annotate("eis 20 FPS", (1, FPS_REQUIREMENT), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=9, color=TEXT)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, 27)
    axes[0][0].set_ylabel("Gemiddelde FPS (beeld tot beslissing)")
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.savefig(FIGURES / "chain_fps.png")
    plt.close(fig)


def plot_chain_breakdown(chain):
    """Waar de tijd per frame naartoe gaat: beeld ophalen, netwerk, server-overhead, inferentie."""
    parts = [("Beeld ophalen (AirSim)", "#b4b2ab"), ("Netwerk", "#eda100"),
             ("Server: JPEG + voor/nabewerking", "#6250d6"), ("Inferentie (TensorRT)", "#2a78d6")]
    keys = [k for k in engines() if k in chain]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(keys))[::-1]
    left = np.zeros(len(keys))
    for (name, color), values in zip(parts, [
            [chain[k]["capture_ms"].mean() for k in keys],
            [chain[k]["network_ms"].mean() for k in keys],
            [chain[k]["server_ms"].mean() - chain[k]["inference_ms"].mean() for k in keys],
            [chain[k]["inference_ms"].mean() for k in keys]]):
        ax.barh(y, values, left=left, color=color, edgecolor=BG, linewidth=2, label=name, height=0.6)
        left += np.array(values)
    for yi, k in zip(y, keys):
        ax.annotate(f"{chain[k]['frame_ms'].mean():.0f} ms", (chain[k]["frame_ms"].mean(), yi),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.axvline(1000 / FPS_REQUIREMENT, color=TEXT, linewidth=1, linestyle="--")
    ax.annotate("50 ms = 20 FPS", (1000 / FPS_REQUIREMENT, len(keys) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=TEXT)
    ax.set_yticks(y, [label(*k) for k in keys])
    ax.set_xlabel("Gemiddelde tijd per frame (ms)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Ketentest: opbouw van de frametijd", loc="left", fontweight="bold")
    fig.savefig(FIGURES / "chain_breakdown.png")
    plt.close(fig)


def plot_inference(rows):
    """Inferentietijd enkel op de Jetson (offline), per engine."""
    values = {(m, p): offline_value(rows, m, p, "all", None, "inference_ms") for m, p in engines()}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    grouped_bars(ax, values, "{:.1f}")
    ax.set_ylabel("Inferentietijd per beeld (ms)")
    ax.legend(loc="upper left")
    ax.set_title("Inferentietijd op de Jetson (15W)", loc="left", fontweight="bold")
    fig.savefig(FIGURES / "inference_ms.png")
    plt.close(fig)


# ---------------------------------------------------------------- live (deelvragen 7 en 8)

def load_live():
    live = {}
    for model, p in engines():
        path = latest(f"live/live_{model}_{p}_*_attempts.csv")
        if path:
            frames_path = path.with_name(path.name.replace("_attempts.csv", "_frames.csv"))
            live[(model, p)] = {"attempts": read_csv(path), "frames": read_csv(frames_path)}
    return live


def plot_live_outcomes(live):
    """Uitkomsten per engine en per starthoek (en zonder platform)."""
    keys = [k for k in engines() if k in live]
    groups = ["0", "15", "30", "45", "geen platform"]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys) + 1, 4.2), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], keys):
        attempts = live[k]["attempts"]
        bottom = np.zeros(len(groups))
        for outcome in OUTCOMES:
            counts = []
            for g in groups:
                sel = [a for a in attempts if (a["pad_present"] == "0") == (g == "geen platform")
                       and (g == "geen platform" or float(a["theta_deg"]) == float(g))]
                counts.append(sum(a["outcome"] == outcome for a in sel))
            if any(counts):
                ax.bar(range(len(groups)), counts, bottom=bottom, color=OUTCOME_COLORS[outcome],
                       edgecolor=BG, linewidth=2, label=outcome.replace("_", " "))
            bottom += counts
        correct = sum(a["outcome"] in CORRECT for a in attempts)
        ax.set_title(f"{label(*k)}\n{correct}/{len(attempts)} correct", loc="left", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(groups)), ["0°", "15°", "30°", "45°", "geen"], fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Aantal pogingen")
    handles = {h.get_label(): h for ax in axes[0] for h in ax.get_legend_handles_labels()[0]}
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Live landingen: uitkomst per starthoek", x=0.02, y=0.99, ha="left", fontweight="bold")
    fig.savefig(FIGURES / "live_outcomes.png")
    plt.close(fig)


def plot_live_confidence(live):
    """Mediane confidence tegenover hoogte tijdens het dalen (pogingen met platform)."""
    bins = np.arange(3, 22, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        for p in PRECISIONS:
            if (model, p) not in live:
                continue
            frames = [f for f in live[(model, p)]["frames"] if f["pad_present"] == "1" and f["detected"] == "1"]
            alt = np.array([float(f["altitude_m"]) for f in frames])
            conf = np.array([float(f["confidence"]) for f in frames])
            centers, medians = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = conf[(alt >= lo) & (alt < hi)]
                if len(sel) >= 20:
                    centers.append((lo + hi) / 2)
                    medians.append(np.median(sel))
            ax.plot(centers, medians, color=PRECISION_COLORS[p], marker=PRECISION_MARKERS[p],
                    markersize=5, linewidth=2, label=p.upper())
        ax.axhline(0.5, color=TEXT, linewidth=1, linestyle="--")
        ax.annotate("drempel 0,5", (1, 0.5), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", fontsize=9)
        ax.set_xlabel("Hoogte boven het platform (m)")
        ax.set_title(MODEL_LABELS[model], loc="left", fontweight="bold")
        ax.invert_xaxis()   # zoals de landing verloopt: van hoog naar laag
    axes[0].set_ylabel("Mediane confidence")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower left")
    fig.subplots_adjust(top=0.85)
    fig.suptitle("Live: confidence tijdens het dalen (blind onder 3 m)", x=0.06, ha="left", fontweight="bold")
    fig.savefig(FIGURES / "live_confidence.png")
    plt.close(fig)


# ---------------------------------------------------------------- samenvatting

def binom_cdf(k, n, p):
    """P(X <= k) voor X ~ Binomiaal(n, p)."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95 %-betrouwbaarheidsinterval voor een kans na k successen op n pogingen.

    Waarom exact (Clopper-Pearson) en niet 'p ± 1,96·sd': bij 50/50 correct is de
    geschatte foutkans 0 en geeft de normale benadering een interval van breedte 0,
    wat niets zegt. Het exacte interval zegt eerlijk hoe zeker 50 pogingen ons maken.
    De grenzen worden gezocht met bisectie op de binomiale verdeling (geen scipy nodig).
    """
    def solve(f):   # f stijgt van negatief naar positief op [0, 1]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: (1 - alpha / 2) - binom_cdf(k - 1, n, p))
    upper = 1.0 if k == n else solve(lambda p: alpha / 2 - binom_cdf(k, n, p))
    return lower, upper


ROBUSTNESS_CONDITIONS = [("clean", "geen (referentie)"), ("fog_05", "mist 0,5"), ("fog_10", "mist 1,0"),
                         ("dust_05", "stof 0,5"), ("dust_10", "stof 1,0"), ("dark", "weinig licht"),
                         ("blur", "bewegingsonscherpte"), ("noise", "ruis")]


def load_robustness():
    path = RESULTS / "robustness_pc" / "robustness_metrics.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_robustness(rows):
    """Recall per afstand en conditie (links), precision en vals alarm over alles (rechts), per model.

    Recall zegt hoe vaak het platform gemist wordt (gevolg: afbreken). Precision en vals
    alarm zeggen hoe vaak het model iets ziet dat er niet is (gevolg: mogelijk foute landing).
    """
    distances = [5, 10, 20, 30]
    conds = [c for c, _ in ROBUSTNESS_CONDITIONS if any(r["condition"] == c for r in rows)]
    names = dict(ROBUSTNESS_CONDITIONS)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), gridspec_kw={"width_ratios": [4, 2, 4, 2]})
    for k, model in enumerate(MODELS):
        ax_r, ax_p = axes[2 * k], axes[2 * k + 1]
        recall = np.full((len(conds), len(distances)), np.nan)
        overall = np.full((len(conds), 2), np.nan)
        for r in rows:
            if r["model"] != model or r["condition"] not in conds:
                continue
            i = conds.index(r["condition"])
            if r["group"] == "distance":
                recall[i, distances.index(int(float(r["value"])))] = float(r["recall"])
            elif r["group"] == "all":
                overall[i] = [float(r["precision"]), float(r["false_alarm_rate"] or "nan")]
        for ax, grid, cmap, labels in [(ax_r, recall, "Blues", distances), (ax_p, overall, "Reds", ["prec.", "vals al."])]:
            if cmap == "Blues":
                ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            else:   # precision: laag = slecht (rood), vals alarm: hoog = slecht
                shade = np.column_stack([1 - grid[:, 0], grid[:, 1] * 5])
                ax.imshow(np.clip(shade, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid[i, j]
                    dark_cell = (v > 0.6) if cmap == "Blues" else ((1 - v > 0.6) if j == 0 else v * 5 > 0.6)
                    text = f"{v:.2f}" if not (cmap == "Reds" and j == 1) else f"{100 * v:.0f} %"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8.5,
                            color="white" if dark_cell else INK_ON_LIGHT)
            ax.set_xticks(range(len(labels)), labels)
            ax.grid(False)
        if k == 0:   # zelfde rijen in beide panelen: namen enkel links
            ax_r.set_yticks(range(len(conds)), [names[c] for c in conds])
        else:
            ax_r.set_yticks([])
        ax_p.set_yticks([])
        ax_r.set_xlabel("Afstand d (m)")
        ax_r.set_title(f"{MODEL_LABELS[model]}: recall", loc="left", fontweight="bold")
        ax_p.set_title("over alles", loc="left", fontsize=9)
    fig.subplots_adjust(wspace=0.08, top=0.84, left=0.12)
    fig.suptitle("Moeilijkere omstandigheden (PyTorch op de pc, zelfde 1000 testposities)",
                 x=0.02, ha="left", fontweight="bold")
    fig.savefig(FIGURES / "robustness.png")
    plt.close(fig)


# Eigen kleuren per conditie (niet de kleuren van de rekenprecisies, om verwarring te vermijden).
# Gecontroleerd op kleurenblindheid; contrast is laag, daarom staat de naam ook bij elke lijn.
SWEEP_CONDITIONS = [("normal", "normaal", "#008300"), ("fog_05", "mist 0,5", "#e87ba4"), ("dark", "weinig licht", "#eda100")]


def plot_threshold_sweep():
    """Recall (vol) en vals alarm (gestreept) tegenover de confidence-drempel, per model."""
    path = RESULTS / "threshold_sweep" / "sweep.csv"
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        for cond, name, color in SWEEP_CONDITIONS:
            sel = sorted((r for r in rows if r["model"] == model and r["condition"] == cond),
                         key=lambda r: float(r["threshold"]))
            if not sel:
                continue
            t = [float(r["threshold"]) for r in sel]
            recall = [float(r["recall"]) for r in sel]
            alarm = [float(r["false_alarm"]) for r in sel]
            ax.plot(t, recall, color=color, linewidth=2, marker="o", markersize=4)
            ax.plot(t, alarm, color=color, linewidth=2, linestyle="--")
            ax.annotate(name, (t[-1], recall[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=TEXT)
        ax.axvline(0.5, color=TEXT, linewidth=1, linestyle=":")
        ax.annotate("drempel landing", (0.5, 1.02), xytext=(3, 0), textcoords="offset points", fontsize=8.5)
        ax.set_xlim(0, 1.12)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Confidence-drempel")
        ax.set_title(MODEL_LABELS[model], loc="left", fontweight="bold")
    axes[0].set_ylabel("Aandeel beelden")
    handles = [plt.Line2D([], [], color=TEXT, linewidth=2, marker="o", markersize=4, label="recall (platform gevonden)"),
               plt.Line2D([], [], color=TEXT, linewidth=2, linestyle="--", label="vals alarm (beelden zonder platform)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(top=0.84, bottom=0.22)
    fig.suptitle("Drempel: recall en vals alarm (PyTorch op de pc)", x=0.06, ha="left", fontweight="bold")
    fig.savefig(FIGURES / "threshold_sweep.png")
    plt.close(fig)


def plot_robustness_precision():
    """Recall per conditie en rekenprecisie, op de Jetson (6 engines). Eén paneel per model."""
    path = RESULTS / "robustness_jetson" / "robustness_metrics.csv"
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["group"] == "all"]
    conds = [c for c, _ in ROBUSTNESS_CONDITIONS if any(r["condition"] == c for r in rows)]
    names = dict(ROBUSTNESS_CONDITIONS)
    names.update(clean="geen", blur="onscherpte")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, sharey=True)
    width = 0.26
    for ax, model in zip(axes, MODELS):
        for i, p in enumerate(PRECISIONS):
            y = []
            for c in conds:
                match = [float(r["recall"]) for r in rows
                         if r["model"] == model and r["precision_mode"] == p and r["condition"] == c]
                y.append(match[0] if match else np.nan)
            x = np.arange(len(conds)) + (i - 1) * (width + 0.02)
            bars = ax.bar(x, y, width, color=PRECISION_COLORS[p], label=p.upper())
            for b, v in zip(bars, y):
                if not np.isnan(v):
                    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=7, color=TEXT)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Recall")
        ax.set_title(MODEL_LABELS[model], loc="left", fontweight="bold")
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(range(len(conds)), [names[c] for c in conds])
    axes[0].legend(loc="upper right", ncol=3)
    fig.subplots_adjust(hspace=0.3, top=0.9)
    fig.suptitle("Moeilijkere omstandigheden op de Jetson: recall per rekenprecisie (1000 beelden per conditie)",
                 x=0.06, ha="left", fontweight="bold")
    fig.savefig(FIGURES / "robustness_precision.png")
    plt.close(fig)


def write_summary(offline, chain, live):
    rows = []
    for model, p in engines():
        row = {"model": model, "precision_mode": p}
        if offline:
            for metric in ["precision", "recall", "map50", "map50_95", "inference_ms", "false_alarm_rate"]:
                row[metric] = offline_value(offline, model, p, "all", None, metric)
            row["map50_95_d30"] = offline_value(offline, model, p, "distance", 30, "map50_95")
            row["map50_95_theta45"] = offline_value(offline, model, p, "theta", 45, "map50_95")
        if (model, p) in chain:
            mean, p5 = fps_stats(chain[(model, p)]["frame_ms"])
            row.update(chain_fps_mean=mean, chain_fps_p5=p5, chain_fps_ok=int(mean >= FPS_REQUIREMENT))
        if (model, p) in live:
            attempts = live[(model, p)]["attempts"]
            frame_ms = np.array([float(f["frame_ms"]) for f in live[(model, p)]["frames"]])
            mean, p5 = fps_stats(frame_ms)
            correct = sum(a["outcome"] in CORRECT for a in attempts)
            ci_low, ci_high = clopper_pearson(correct, len(attempts))
            row.update(live_attempts=len(attempts), live_fps_mean=mean, live_fps_p5=p5,
                       live_correct_pct=100 * correct / len(attempts),
                       live_correct_ci95_low_pct=100 * ci_low, live_correct_ci95_high_pct=100 * ci_high)
            for outcome in OUTCOMES:
                row[outcome] = sum(a["outcome"] == outcome for a in attempts)
        rows.append(row)
    columns = list(dict.fromkeys(c for r in rows for c in r))
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def apply_dark_theme():
    """Donkere versie voor de presentatiewebsite (cockpitstijl). Enkel kleuren veranderen, de data niet."""
    global TEXT, MUTED, GRID, BG, FIGURES
    TEXT, MUTED, GRID, BG = "#e6f2ec", "#8fa59d", "#24343a", "#152024"
    FIGURES = RESULTS / "figures_dark"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": TEXT,
        "axes.titlecolor": TEXT, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "legend.labelcolor": TEXT,
    })


def main():
    if "--dark" in sys.argv:
        apply_dark_theme()
    FIGURES.mkdir(parents=True, exist_ok=True)
    offline, chain, live = load_offline(), load_chain(), load_live()
    print(f"Gevonden: offline {'ja' if offline else 'nee'}, keten {len(chain)}/6, live {len(live)}/6 engines")
    if offline:
        plot_offline_per_group(offline)
        plot_offline_heatmaps(offline)
        plot_inference(offline)
    live_fps = {k: fps_stats(np.array([float(f["frame_ms"]) for f in v["frames"]]))[0] for k, v in live.items()}
    if chain:
        plot_chain_fps(chain, live_fps)
        plot_chain_breakdown(chain)
    if live:
        plot_live_outcomes(live)
        plot_live_confidence(live)
    robustness = load_robustness()
    if robustness:
        plot_robustness(robustness)
    plot_threshold_sweep()
    plot_robustness_precision()
    rows = write_summary(offline, chain, live)

    print(f"\n{'engine':18s} {'mAP50-95':>8s} {'inf ms':>7s} {'keten':>6s} {'p5':>5s} {'live':>6s} {'correct':>8s} {'95%-BI':>14s}")
    for r in rows:
        def fmt(k, spec):
            return format(r[k], spec) if k in r else format("-", spec[0] + "s")
        print(f"{label(r['model'], r['precision_mode']):18s} {fmt('map50_95', '8.3f')} {fmt('inference_ms', '7.1f')} "
              f"{fmt('chain_fps_mean', '6.1f')} {fmt('chain_fps_p5', '5.1f')} {fmt('live_fps_mean', '6.1f')} "
              f"{fmt('live_correct_pct', '7.0f')} % {fmt('live_correct_ci95_low_pct', '6.1f')}-{fmt('live_correct_ci95_high_pct', '5.1f')} %")
    print(f"\nGrafieken: {FIGURES}\nTabel:     {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
