"""Ejecuta todo el pipeline de seguido:
entrenar (baseline y curriculum) -> evaluar -> gráficas.

Uso:
    python main.py            # entrenamiento completo (100.000 pasos)
    python main.py --rapido   # prueba corta (5.000 pasos) para comprobar que todo funciona

TensorBoard se abre aparte:  tensorboard --logdir logs
"""
import argparse
import subprocess
import sys


ALGORITMOS = ["ppo", "a2c", "dqn", "qrdqn"]
SEMILLAS = [0, 1, 2]
VARIANTES = (",".join(f"baseline=models/{{algo}}_s{s}" for s in SEMILLAS) + "," +
             ",".join(f"curriculum=models/{{algo}}_curriculum_s{s}" for s in SEMILLAS))


def ejecutar(comando):
    """Lanza un comando y para el programa si da error."""
    print("\n>>>", " ".join(comando))
    subprocess.run(comando, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true",
                    help="Prueba corta: 5.000 pasos")
    args = ap.parse_args()

    if args.rapido:
        pasos, eval_freq, episodios = "5000", "1000", "10"
    else:
        pasos, eval_freq, episodios = "1000000", "100000", "100"

    python = sys.executable  # el mismo Python (y entorno) con el que se lanza main.py

    # 1) Entrenar los 4 algoritmos, sin curriculum y con curriculum
    for algo in ALGORITMOS:
        for seed in SEMILLAS:
            for extra in ([], ["--curriculum"]):
                ejecutar([python, f"training_{algo}.py", "--timesteps", pasos,
                        "--eval-freq", eval_freq, "--seed", str(seed)] + extra)

    # 2) Evaluar los mejores modelos y guardar el CSV
    ejecutar([python, "evaluate_best_to_csv.py",
              "--episodes", episodios, "--variants", VARIANTES])

    # 3) Gráfica comparativa a partir del CSV
    ejecutar([python, "plot_comparison.py"])

    # 4) Benchmark episodio a episodio (manzanas)
    # ejecutar([python, "benchmark.py", "--all", "--episodes", episodios,
    #           "--metric", "apples", "--variants", VARIANTES])

    print("\nTerminado. Resultados en logs/ y modelos en models/.")
    print("Para ver las curvas:  tensorboard --logdir logs")


if __name__ == "__main__":
    main()