"""PPO con 1.000.000 de pasos: entrena, evalúa y dibuja, todo seguido.

Todo se guarda en una carpeta aparte (models/ y logs/ dentro de ella),
así que NO se sobrescriben los resultados de 200.000 pasos.

Uso:
    uv run python main_ppo.py --rapido   # prueba corta (5.000 pasos), carpeta aparte
    uv run python main_ppo.py            # entrenamiento completo (1.000.000 pasos)

TensorBoard:
    uv run tensorboard --logdir resultados_PPO_1M/logs
"""
import argparse
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent  # carpeta del proyecto, donde están los scripts
SEMILLAS = [0, 1, 2]

VARIANTES = ",".join(
    [f"baseline=models/{{algo}}_s{s}" for s in SEMILLAS]
    + [f"curriculum=models/{{algo}}_curriculum_s{s}" for s in SEMILLAS]
)


def ejecutar(script, argumentos, carpeta):
    """Lanza un script del proyecto trabajando dentro de 'carpeta'.

    Los scripts guardan en models/ y logs/ relativos a donde se ejecutan,
    así que todo acaba dentro de 'carpeta'. Para si hay un error.
    """
    comando = [sys.executable, str(AQUI / script)] + argumentos
    print("\n>>>", script, " ".join(argumentos))
    subprocess.run(comando, check=True, cwd=carpeta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true",
                    help="Prueba corta: 5.000 pasos, en otra carpeta")
    ap.add_argument("--pasos", type=int, default=1_000_000)
    args = ap.parse_args()

    if args.rapido:
        pasos, eval_freq, episodios = 5000, 1000, 10
        nombre_carpeta = "resultados_PPO_prueba"
    else:
        pasos, eval_freq, episodios = args.pasos, args.pasos // 20, 100
        nombre_carpeta = "resultados_PPO_1M"

    carpeta = AQUI / nombre_carpeta
    carpeta.mkdir(exist_ok=True)
    print(f"Resultados en: {carpeta}")

    # 1) Entrenar PPO: 3 semillas, sin curriculum y con curriculum
    for seed in SEMILLAS:
        for extra in ([], ["--curriculum"]):
            ejecutar("training_ppo.py",
                     ["--timesteps", str(pasos), "--eval-freq", str(eval_freq),
                      "--seed", str(seed)] + extra, carpeta)

    # 2) Evaluar los mejores modelos y guardar el CSV
    ejecutar("evaluate_best_to_csv.py",
             ["--algorithm", "ppo", "--episodes", str(episodios),
              "--csv-out", "logs/ppo_evaluation.csv",
              "--variants", VARIANTES], carpeta)

    # 3) Gráfica comparativa a partir del CSV
    ejecutar("plot_comparison.py",
             ["--csv", "logs/ppo_evaluation.csv",
              "--out", "logs/ppo_comparison.png"], carpeta)

    # 4) Benchmark del challenge (semilla 0), manzanas por episodio
    ejecutar("benchmark.py",
             ["--algorithm", "ppo", "--episodes", str(episodios),
              "--metric", "apples",
              "--variants",
              "baseline=models/{algo}_s0,curriculum=models/{algo}_curriculum_s0"],
             carpeta)

    print(f"\nTerminado. Todo está en: {carpeta}")
    print(f"Curvas:  uv run tensorboard --logdir {nombre_carpeta}/logs")


if __name__ == "__main__":
    main()