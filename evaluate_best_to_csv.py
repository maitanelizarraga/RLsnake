import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import QRDQN

import snake_env  # registra Snake-v0

ALGORITHMS = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "QRDQN": QRDQN}

DEFAULT_EPISODES = 100
DEFAULT_CSV = "logs/best_models_evaluation.csv"
FIELDNAMES = ["algorithm", "episode", "return", "apples", "steps", "terminated", "truncated"]


def resolve_best_model_path(algo_upper: str) -> Path | None:
    """Busca el best_model.zip (o en su defecto el checkpoint numerico) de un algoritmo."""
    candidates = [
        Path(f"models/{algo_upper}/best_model/best_model.zip"),  # EvalCallback
        Path(f"models/{algo_upper}/best_model"),                 # por si es carpeta
        Path(f"models/{algo_upper}/100000.zip"),                 # model.save(...)
        Path(f"models/{algo_upper}/100000"),
    ]
    for c in candidates:
        if c.is_file():
            return c
        if c.is_dir():
            inner = c / "best_model.zip"
            if inner.exists():
                return inner
    return None


def run_episodes(model, env, num_episodes: int, deterministic: bool = True) -> list[dict]:
    rows = []
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total, steps = 0.0, 0
        terminated = truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        rows.append({
            "episode": ep,
            "return": total,
            "apples": info.get("apples", ""),
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evalua el best_model de cada algoritmo y guarda un CSV combinado. "
            "Sin argumentos: los 4 algoritmos, 100 episodios cada uno."
        )
    )
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                    help=f"Episodios por algoritmo (defecto: {DEFAULT_EPISODES}).")
    ap.add_argument("--algorithm", choices=[a.lower() for a in ALGORITHMS],
                    default=None, help="Evaluar solo este algoritmo (defecto: todos).")
    ap.add_argument("--env-id", default="Snake-v0")
    ap.add_argument("--csv-out", default=DEFAULT_CSV,
                    help=f"CSV de salida (defecto: {DEFAULT_CSV}).")
    ap.add_argument("--stochastic", action="store_true",
                    help="Politica no determinista (defecto: determinista).")
    args = ap.parse_args()

    algos = [args.algorithm.upper()] if args.algorithm else list(ALGORITHMS)
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array")
    all_rows: list[dict] = []
    summary: list[tuple[str, float, float, float, float]] = []

    try:
        for algo_upper in algos:
            model_path = resolve_best_model_path(algo_upper)
            if model_path is None:
                print(f"[WARN] {algo_upper}: no se encontro modelo en models/{algo_upper}/ (skip)")
                continue

            print(f"\n=== {algo_upper} -> {model_path} ({args.episodes} episodios) ===")
            try:
                model = ALGORITHMS[algo_upper].load(str(model_path))
                rows = run_episodes(model, env, args.episodes,
                                    deterministic=not args.stochastic)
            except Exception as exc:
                print(f"[ERROR] {algo_upper}: {exc}")
                continue

            for r in rows:
                r["algorithm"] = algo_upper
            all_rows.extend(rows)

            ret = np.array([r["return"] for r in rows], dtype=float)
            summary.append((algo_upper, ret.mean(), ret.std(), ret.max(), ret.min()))
    finally:
        env.close()

    if not all_rows:
        print("\nNo se evaluo ningun modelo (¿faltan best_model.zip en models/?).")
        return

    # --- CSV combinado ---
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})

    print(f"\nCSV combinado guardado en: {csv_out} ({len(all_rows)} filas)")

    # --- Resumen por consola ---
    print("\nResumen:")
    print(f"{'algo':<6} {'mean':>8} {'std':>8} {'max':>8} {'min':>8}")
    for algo, m, s, mx, mn in summary:
        print(f"{algo:<6} {m:>8.2f} {s:>8.2f} {mx:>8.2f} {mn:>8.2f}")


if __name__ == "__main__":
    main()