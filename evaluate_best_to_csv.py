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
DEFAULT_VARIANTS = "baseline=models/{algo}"

FIELDNAMES = [
    "variant", "algorithm", "episode",
    "return", "apples", "steps", "apples_per_100_steps",
    "terminated", "truncated",
]


def parse_variants(spec: str) -> list[tuple[str, str]]:
    """Parse 'name=template,name=template' -> [(name, template), ...]."""
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


def resolve_best_model_path(base_dir: Path) -> Path | None:
    candidates = [
        base_dir / "best_model" / "best_model.zip",
        base_dir / "best_model",
        base_dir / "100000.zip",
        base_dir / "100000",
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

        apples = info.get("apples", 0)
        rows.append({
            "episode": ep,
            "return": total,
            "apples": apples,
            "steps": steps,
            "apples_per_100_steps": 100.0 * float(apples) / max(steps, 1),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evalúa los best_model de cada algoritmo/variante y guarda un CSV combinado."
    )
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--algorithm", choices=[a.lower() for a in ALGORITHMS], default=None)
    ap.add_argument("--env-id", default="Snake-v0")
    ap.add_argument("--csv-out", default=DEFAULT_CSV)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help="Lista nombre=plantilla. Plantilla usa {algo}. "
                         "Ejemplo: 'baseline=models/{algo},curriculum=models/{algo}_curriculum'")
    args = ap.parse_args()

    variants = parse_variants(args.variants)
    algos = [args.algorithm.upper()] if args.algorithm else list(ALGORITHMS)
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array", curriculum_level=4)
    all_rows: list[dict] = []
    summary: list[tuple[str, str, float, float, float, float, float]] = []

    try:
        for variant_name, template in variants:
            for algo_upper in algos:
                base_dir = Path(template.format(algo=algo_upper))
                model_path = resolve_best_model_path(base_dir)
                if model_path is None:
                    print(f"[WARN] {variant_name}/{algo_upper}: no model under "
                          f"'{base_dir}' (skip)")
                    continue

                print(f"\n=== {variant_name} / {algo_upper} -> {model_path} "
                      f"({args.episodes} episodios) ===")
                try:
                    model = ALGORITHMS[algo_upper].load(str(model_path))
                    rows = run_episodes(model, env, args.episodes,
                                        deterministic=not args.stochastic)
                except Exception as exc:
                    print(f"[ERROR] {variant_name}/{algo_upper}: {exc}")
                    continue

                for r in rows:
                    r["variant"] = variant_name
                    r["algorithm"] = algo_upper
                all_rows.extend(rows)

                ret = np.array([r["return"] for r in rows], dtype=float)
                eff = np.array([r["apples_per_100_steps"] for r in rows], dtype=float)
                summary.append((
                    variant_name, algo_upper,
                    ret.mean(), ret.std(), ret.max(), ret.min(), eff.mean(),
                ))
    finally:
        env.close()

    if not all_rows:
        print("\nNo se evaluó ningún modelo.")
        return

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})

    print(f"\nCSV combinado guardado en: {csv_out} ({len(all_rows)} filas)")
    print("\nResumen:")
    print(f"{'variant':<12} {'algo':<6} {'mean':>8} {'std':>8} "
          f"{'max':>8} {'min':>8} {'apples/100':>12}")
    for v, a, m, s, mx, mn, ef in summary:
        print(f"{v:<12} {a:<6} {m:>8.2f} {s:>8.2f} {mx:>8.2f} {mn:>8.2f} {ef:>12.2f}")


if __name__ == "__main__":
    main()