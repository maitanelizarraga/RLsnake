import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import QRDQN

import snake_env  # noqa: F401

ALGORITHMS = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "qrdqn": QRDQN}

METRICS = {
    "reward": {"label": "retorno", "unit": "puntos"},
    "apples": {"label": "manzanas recogidas", "unit": "manzanas"},
}

DEFAULT_VARIANTS = "baseline=models/{algo}"



def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.exists():
        return path
    zip_path = path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError(f"No se encontró '{model_path}' ni '{zip_path}'.")


def resolve_model_in_dir(base_dir: Path) -> Path | None:
    candidates = [
        base_dir / "best_model" / "best_model.zip",
        base_dir / "best_model",
        base_dir / "100000.zip",
        base_dir / "100000",
    ]
    for c in candidates:
        if c.is_file():
            return c
        if c.is_dir() and (c / "best_model.zip").exists():
            return c / "best_model.zip"
    return None


def parse_variants(spec: str) -> list[tuple[str, str]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Variante inválida (falta '='): {item!r}")
        name, tmpl = item.split("=", 1)
        out.append((name.strip(), tmpl.strip()))
    if not out:
        raise ValueError("No se ha definido ninguna variante.")
    return out


# ---------------------------------------------------------------- eval

def run_episode(model, env, metric: str, deterministic: bool = True) -> tuple[float, int]:
    obs, info = env.reset()
    done = False
    total = 0.0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        done = terminated or truncated
    value = total if metric == "reward" else float(info.get("apples", 0))
    return value, steps


def evaluate_policy_metric(model, env, metric: str, num_episodes: int = 100,
                           deterministic: bool = True) -> np.ndarray:
    values = []
    for episode in range(1, num_episodes + 1):
        value, steps = run_episode(model, env, metric=metric,
                                   deterministic=deterministic)
        values.append(value)
        print(f"Episodio {episode:03d} | {METRICS[metric]['label']} = {value:.2f} "
              f"| steps = {steps}")
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------- plots

def plot_metric(values: np.ndarray, metric: str, output_path: Path,
                title_prefix: str = "", show_plot: bool = False) -> None:
    episodes = np.arange(1, len(values) + 1)
    max_idx, min_idx = int(np.argmax(values)), int(np.argmin(values))
    colors = np.array(["tab:blue"] * len(values), dtype=object)
    colors[max_idx] = colors[min_idx] = "red"
    mean_value = float(np.mean(values))
    label = METRICS[metric]["label"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(episodes, values, color=colors, edgecolor="black", linewidth=0.2)
    ax.axhline(mean_value, color="black", linestyle="--", linewidth=2,
               label=f"{label.capitalize()} medio = {mean_value:.2f}")
    ax.set_title(f"Benchmark de {label} por episodio ({len(values)} episodios){title_prefix}")
    ax.set_xlabel("Episodio")
    ax.set_ylabel(label.capitalize())
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")

    if max_idx != min_idx:
        ax.text(episodes[max_idx], values[max_idx], f"MAX {values[max_idx]:.2f}",
                color="red", fontsize=9, ha="center", va="bottom")
        ax.text(episodes[min_idx], values[min_idx], f"MIN {values[min_idx]:.2f}",
                color="red", fontsize=9, ha="center", va="top")
    else:
        ax.text(episodes[max_idx], values[max_idx], f"= {values[max_idx]:.2f}",
                color="red", fontsize=9, ha="center", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Gráfica guardada en: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_comparison(results: dict[tuple[str, str], np.ndarray], metric: str,
                    output_path: Path, show_plot: bool = False) -> None:
    """Barras agrupadas por variante. results[(variant, ALGO)] = array."""
    if not results:
        return
    algos = sorted({algo for _, algo in results})
    variants = sorted({v for v, _ in results})
    n_v = len(variants)
    width = 0.8 / n_v
    x = np.arange(len(algos))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_v, 1)))
    label = METRICS[metric]["label"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, variant in enumerate(variants):
        means, stds = [], []
        for algo in algos:
            arr = results.get((variant, algo))
            means.append(float(np.mean(arr)) if arr is not None else 0.0)
            stds.append(float(np.std(arr)) if arr is not None else 0.0)
        offset = (i - (n_v - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=variant, color=colors[i], edgecolor="black")
        for xi, (m, s) in zip(x + offset, zip(means, stds)):
            ax.text(xi, m + s, f"{m:.2f}\n±{s:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel(label.capitalize())
    ep_count = len(next(iter(results.values())))
    ax.set_title(f"Comparación de {label} medio por algoritmo/variante "
                 f"({ep_count} episodios)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if n_v > 1:
        ax.legend(title="variante")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Gráfica comparativa guardada en: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark de políticas entrenadas en Snake.")
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="ppo")
    parser.add_argument("--all", action="store_true",
                        help="Evaluar todos los algoritmos (ignora --algorithm/--model-path).")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS,
                        help="Lista nombre=plantilla. Plantilla usa {algo}. "
                             "Ejemplo: 'baseline=models/{algo},curriculum=models/{algo}_curriculum'")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Solo válido con --algorithm y una única --variants.")
    parser.add_argument("--env-id", type=str, default="Snake-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--metric", choices=list(METRICS), default="reward")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-plot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    deterministic = not args.stochastic
    metric = args.metric

    variants = parse_variants(args.variants)
    algorithms = list(ALGORITHMS) if args.all else [args.algorithm]

    if args.model_path and (args.all or len(variants) > 1):
        print("[WARN] --model-path se ignora porque hay varias variantes o --all.")

    results: dict[tuple[str, str], np.ndarray] = {}

    for variant_name, template in variants:
        for algo in algorithms:
            algo_upper = algo.upper()

            if args.model_path and not args.all and len(variants) == 1:
                model_path = resolve_model_path(args.model_path)
            else:
                base_dir = Path(template.format(algo=algo_upper))
                model_path = resolve_model_in_dir(base_dir)
                if model_path is None:
                    print(f"[WARN] {variant_name}/{algo_upper}: sin modelo en "
                          f"'{base_dir}' (skip)")
                    if args.all:
                        continue
                    raise FileNotFoundError(
                        f"No hay modelo en {base_dir} ni subcarpetas conocidas."
                    )

            print(f"\n=== {variant_name} / {algo_upper} -> '{model_path}' ===")
            model = ALGORITHMS[algo].load(model_path)
            env = gym.make(args.env_id, render_mode="rgb_array", curriculum_level=4)
            try:
                values = evaluate_policy_metric(model, env, metric=metric,
                                                num_episodes=args.episodes,
                                                deterministic=deterministic)
            finally:
                env.close()
            results[(variant_name, algo_upper)] = values

            if args.all:
                per_path = Path(f"logs/benchmark_{variant_name}_{algo_upper}_{metric}.png")
                plot_metric(values, metric=metric, output_path=per_path,
                            title_prefix=f" — {variant_name} / {algo_upper}",
                            show_plot=False)

    if not results:
        print("No se pudo evaluar ningún modelo.")
        return

    label = METRICS[metric]["label"]
    print("\nResumen benchmark")
    for (variant, algo), values in results.items():
        print(f"[{variant}/{algo}] {label} medio: {np.mean(values):.2f} | "
              f"max: {np.max(values):.2f} | min: {np.min(values):.2f}")

    if args.all or len(results) > 1:
        output = Path(args.plot_path or f"logs/benchmark_comparison_{metric}.png")
        plot_comparison(results, metric=metric, output_path=output,
                        show_plot=args.show_plot)
    else:
        (variant, algo), values = next(iter(results.items()))
        output = Path(args.plot_path or f"logs/benchmark_{variant}_{algo}_{metric}.png")
        plot_metric(values, metric=metric, output_path=output,
                    title_prefix=f" — {variant} / {algo}",
                    show_plot=args.show_plot)


if __name__ == "__main__":
    main()