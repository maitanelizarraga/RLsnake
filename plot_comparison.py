"""plot_comparison.py — Comparativa visual de los 4 modelos en una figura 2x2."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = "logs/best_models_evaluation.csv"
DEFAULT_OUT = "logs/models_comparison.png"


def load_csv(path: Path) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            algo = row["algorithm"]
            for col in ("return", "apples", "steps"):
                val = row.get(col, "")
                if val == "":
                    continue
                try:
                    data[algo][col].append(float(val))
                except ValueError:
                    pass
    return data


def plot(data: dict, output: Path, show: bool = False) -> None:
    algos = sorted(data)
    if not algos:
        print("CSV vacío, nada que graficar.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def bar(ax, metric: str, title: str, ylabel: str) -> None:
        means = [np.mean(data[a][metric]) if data[a][metric] else 0.0 for a in algos]
        stds = [np.std(data[a][metric])  if data[a][metric] else 0.0 for a in algos]
        ax.bar(algos, means, yerr=stds, capsize=8, color=colors, edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s, f"{m:.2f}\n±{s:.2f}",
                    ha="center", va="bottom", fontsize=9)

    bar(axes[0, 0], "return", "Retorno medio por algoritmo", "Retorno")
    bar(axes[0, 1], "apples", "Manzanas medias por algoritmo", "Manzanas")
    bar(axes[1, 0], "steps",  "Pasos medios por algoritmo",    "Pasos")

    # Boxplot de retornos
    ax = axes[1, 1]
    box_data = [data[a]["return"] for a in algos]
    bp = ax.boxplot(box_data, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticks(range(1, len(algos) + 1))
    ax.set_xticklabels(algos)
    ax.set_title("Distribución del retorno por algoritmo")
    ax.set_ylabel("Retorno")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Comparación de los 4 modelos entrenados en Snake",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Gráfico comparativo guardado en: {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparativa de los 4 modelos desde el CSV.")
    ap.add_argument("--csv", default=DEFAULT_CSV, help=f"CSV de entrada (defecto: {DEFAULT_CSV})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"PNG de salida (defecto: {DEFAULT_OUT})")
    ap.add_argument("--show", action="store_true", help="Mostrar la figura al final")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el CSV: {csv_path}")
        print("Ejecuta primero:  python evaluate_best_to_csv.py")
        return

    plot(load_csv(csv_path), Path(args.out), show=args.show)


if __name__ == "__main__":
    main()