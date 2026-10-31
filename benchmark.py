import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import QRDQN

import snake_env  # noqa: F401 - Registers Snake environments via gymnasium.register

# Mapeo algoritmo -> clase de modelo (todos comparten la misma API .load / .predict)
ALGORITHMS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "qrdqn": QRDQN,
}

METRICS = {
    "reward":  {"label": "retorno", "unit": "puntos"},
    "apples":  {"label": "manzanas recogidas", "unit": "manzanas"},
    "efficiency": {  # ← NUEVA
        "label": "retorno por paso",
        "unit": "puntos/paso",
        "compute": lambda ret, steps: ret / max(steps, 1),
    },
}


def resolve_model_path(model_path: str) -> Path:
    """Resolve model path allowing both with and without .zip extension."""
    path = Path(model_path)
    if path.exists():
        return path
    zip_path = path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError(f"No se encontro el modelo en '{model_path}' ni en '{zip_path}'.")


def run_episode(model, env: gym.Env, metric: str, deterministic: bool = True) -> float:
    obs, info = env.reset()
    done = False
    episode_return = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        done = terminated or truncated
    return episode_return if metric == "reward" else float(info["apples"])


def evaluate_policy_metric(model, env: gym.Env, metric: str, num_episodes: int = 100,
                           deterministic: bool = True) -> np.ndarray:
    values = []
    for episode in range(1, num_episodes + 1):
        value = run_episode(model, env, metric=metric, deterministic=deterministic)
        values.append(value)
        print(f"Episodio {episode:03d} | {METRICS[metric]['label']} = {value:.2f}")
    return np.array(values, dtype=np.float64)


def plot_metric(values: np.ndarray, metric: str, output_path: Path, title_prefix: str = "",
                show_plot: bool = False) -> None:
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
    ax.text(episodes[max_idx], values[max_idx], f"MAX {values[max_idx]:.2f}", color="red",
            fontsize=9, ha="center", va="bottom")
    ax.text(episodes[min_idx], values[min_idx], f"MIN {values[min_idx]:.2f}", color="red",
            fontsize=9, ha="center", va="top")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Grafica guardada en: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_comparison(results: dict[str, np.ndarray], metric: str, output_path: Path,
                    show_plot: bool = False) -> None:
    """Grafica de barras con media ± std por algoritmo."""
    labels = list(results)
    means = np.array([np.mean(results[a]) for a in labels])
    stds = np.array([np.std(results[a]) for a in labels])
    label = METRICS[metric]["label"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=8, color="tab:blue", edgecolor="black")
    ax.set_xticks(x, [a.upper() for a in labels])
    ax.set_ylabel(label.capitalize())
    ax.set_title(f"Comparacion de {label} medio por algoritmo ({len(results[labels[0]])} episodios)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.2f}\n±{s:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Grafica comparativa guardada en: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark de politicas entrenadas en Snake.")
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="ppo",
                        help="Algoritmo a evaluar (ppo, a2c, dqn, qrdqn).")
    parser.add_argument("--all", action="store_true",
                        help="Evaluar todos los algoritmos y generar comparacion (ignora --algorithm/--model-path).")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Ruta al modelo de SB3. Si no se indica se usa models/{ALGO}/100000.")
    parser.add_argument("--env-id", type=str, default="Snake-v0",
                        help="ID del entorno registrado en gymnasium.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Numero de episodios de evaluacion.")
    parser.add_argument("--metric", choices=list(METRICS), default="reward",
                        help="Metrica a evaluar: 'reward' o 'apples'.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Usa politica no determinista (por defecto determinista).")
    parser.add_argument("--plot-path", type=str, default=None,
                        help="Ruta donde guardar la grafica. Por defecto: logs/benchmark_{algo}_{metric}.png "
                             "o logs/benchmark_comparison_{metric}.png cuando se usa --all.")
    parser.add_argument("--show-plot", action="store_true",
                        help="Mostrar la grafica en pantalla al finalizar.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    deterministic = not args.stochastic
    metric = args.metric

    algorithms = list(ALGORITHMS) if args.all else [args.algorithm]
    results: dict[str, np.ndarray] = {}

    for algo in algorithms:
        algo_upper = algo.upper()

        # Ruta del modelo: la indicada por el usuario (solo en modo individual)
        # o el patrón por defecto models/<ALGO>/100000.
        if args.model_path and not args.all:
            model_path_str = args.model_path
        else:
            model_path_str = f"models/{algo_upper}/100000"

        print(f"\n=== Evaluando {algo_upper} desde '{model_path_str}' ===")
        try:
            model_path = resolve_model_path(model_path_str)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            if args.all:
                continue  # si falta uno, seguimos con los demás
            raise

        model = ALGORITHMS[algo].load(model_path)
        env = gym.make(args.env_id, render_mode="rgb_array")
        try:
            values = evaluate_policy_metric(model, env, metric=metric,
                                            num_episodes=args.episodes,
                                            deterministic=deterministic)
        finally:
            env.close()
        results[algo_upper] = values

        # En modo --all, guardamos una grafica individual por algoritmo
        if args.all:
            per_model_path = Path(f"logs/benchmark_{algo_upper}_{metric}.png")
            plot_metric(values, metric=metric, output_path=per_model_path,
                        title_prefix=f" — {algo_upper}", show_plot=False)

    if not results:
        print("No se pudo evaluar ningun modelo (¿faltan los .zip en models/?).")
        return

    # Resumen por consola
    label = METRICS[metric]["label"]
    print("\nResumen benchmark")
    for algo, values in results.items():
        print(f"[{algo}] {label} medio: {np.mean(values):.2f} | "
              f"max: {np.max(values):.2f} | min: {np.min(values):.2f}")

    # Grafica final
    if args.all:
        output = Path(args.plot_path or f"logs/benchmark_comparison_{metric}.png")
        plot_comparison(results, metric=metric, output_path=output, show_plot=args.show_plot)
    else:
        algo_upper = next(iter(results))
        output = Path(args.plot_path or f"logs/benchmark_{algo_upper}_{metric}.png")
        plot_metric(results[algo_upper], metric=metric, output_path=output,
                    title_prefix=f" — {algo_upper}", show_plot=args.show_plot)


if __name__ == "__main__":
    main()