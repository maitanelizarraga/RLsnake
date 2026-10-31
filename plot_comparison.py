"""plot_comparison.py — Comparativa visual agrupada por variante."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = "logs/best_models_evaluation.csv"
DEFAULT_OUT = "logs/models_comparison.png"

CSV_COLUMNS = ("return", "apples", "steps", "apples_per_100_steps")


def load_csv(path: Path):
    """data[(algo, variant)][metric] = list[float]."""
    data: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            algo = row["algorithm"]
            variant = (row.get("variant") or "baseline").strip()
            key = (algo, variant)
            for col in CSV_COLUMNS:
                val = row.get(col, "")
                if val == "":
                    continue
                try:
                    data[key][col].append(float(val))
                except ValueError:
                    pass
    return data


def _grouped_bars(ax, data, algos, variants, metric, title, ylabel, colors,
                  fmt="mean±std"):
    n_v = len(variants)
    width = 0.8 / n_v
    x = np.arange(len(algos))
    for i, variant in enumerate(variants):
        means, stds = [], []
        for algo in algos:
            vals = data.get((algo, variant), {}).get(metric, [])
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        offset = (i - (n_v - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=variant, color=colors[i], edgecolor="black")
        if n_v <= 2:
            for xi, (m, s) in zip(x + offset, zip(means, stds)):
                ax.text(xi, m + s, f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")


def _grouped_boxplot(ax, data, algos, variants, metric, title, ylabel, colors):
    n_v = len(variants)
    positions, box_data, box_colors = [], [], []
    tick_pos, tick_labels = [], []
    for ai, algo in enumerate(algos):
        for vi, variant in enumerate(variants):
            pos = ai * (n_v + 1) + vi
            vals = data.get((algo, variant), {}).get(metric, [])
            if vals:
                positions.append(pos)
                box_data.append(vals)
                box_colors.append(colors[vi])
        tick_pos.append(ai * (n_v + 1) + (n_v - 1) / 2.0)
        tick_labels.append(algo)
    if not box_data:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                transform=ax.transAxes)
        return
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.7)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def plot(data, output: Path, show: bool = False) -> None:
    if not data:
        print("CSV vacío, nada que graficar.")
        return

    algos = sorted({k[0] for k in data})
    variants = sorted({k[1] for k in data})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _grouped_bars(axes[0, 0], data, algos, variants, "return",
                  "Retorno medio por algoritmo/variante", "Retorno", colors)
    _grouped_bars(axes[0, 1], data, algos, variants, "apples",
                  "Manzanas medias por algoritmo/variante", "Manzanas", colors)
    _grouped_bars(axes[1, 0], data, algos, variants, "apples_per_100_steps",
                  "Eficiencia: manzanas por 100 pasos", "Manzanas / 100 pasos",
                  colors)
    _grouped_boxplot(axes[1, 1], data, algos, variants, "return",
                     "Distribución del retorno", "Retorno", colors)

    subtitle = ("Comparación de variantes: " + ", ".join(variants)
                if len(variants) > 1 else f"Variante: {variants[0]}")
    fig.suptitle(f"Comparación de modelos entrenados en Snake\n{subtitle}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Gráfico comparativo guardado en: {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparativa de modelos desde el CSV.")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el CSV: {csv_path}")
        print("Ejecuta primero:  python evaluate_best_to_csv.py "
              "--variants \"baseline=models/{algo},curriculum=models/{algo}_curriculum\"")
        return

    plot(load_csv(csv_path), Path(args.out), show=args.show)


if __name__ == "__main__":
    main()