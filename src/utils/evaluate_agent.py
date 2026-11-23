
"""
evaluate.py
-----------
Valutazione di un singolo modello DQN.
"""

import csv
from pathlib import Path
import argparse

import numpy as np

from core.environment import Environment
from core.agent import DQNAgent, get_max_state_time
from utils.train_agent import calculate_rewards, get_training_goals

PROJECT_DIR=Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR=PROJECT_DIR/ "data"/"output"
EVALUATION_DIR=OUTPUT_DIR/"evaluation"

SEED:54

def _run_evaluation_episode(env, agent, tls_id, goal, max_steps):
    """
    Esegue un episodio di valutazione per un singolo agente.

    Args:
        agent: agente addestrato
        max_steps: numero massimo di step per episodio.
        tls_id: identificativo del semaforo.
        goal: goal utilizzato per calcolare il reward.

    Returns:
        Dizionario contenente le metriche medie.
    """

    agents = {
        tls_id: {
            "agent": agent,
            "goal": goal
        }
    }

    observations = env.start_simulation(
        SEED,
        max_steps=max_steps
    )

    total_reward = 0.0
    steps = 0

    try:
        for step in range(max_steps):

            # Selezione dell'azione senza esplorazione
            action = agent.select_action(
                observations[tls_id],
                explore=False
            )

            actions = {
                tls_id: action
            }

            unsafe_tls = env.get_unsafe_tls(actions)

            next_observations, terminated = env.step(actions)

            collision_tls = env.get_collisions_tls()

            rewards = calculate_rewards(
                env,
                collision_tls,
                unsafe_tls,
                agents
            )

            total_reward += rewards[tls_id]

            observations = next_observations
            steps += 1

            if terminated:
                break

        return {
            "reward": total_reward,
            "steps": steps
        }

    finally:
        env.end_simulation()


def evaluate(model_path, episodes, max_steps, tls_id, goal):
    """
    Valuta un singolo modello DQN.

    Args:
        model_path: percorso del modello da valutare.
        episodes: numero di episodi di valutazione.
        max_steps: numero massimo di step per episodio.
        tls_id: identificativo del semaforo.
        goal: goal utilizzato per calcolare il reward.

    Returns:
        Dizionario contenente le metriche medie.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}"
        )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    env = Environment(
        traffic_lights=[tls_id],
        max_elapsed_time=get_max_state_time()
    )

    agent = DQNAgent(
        link_ids=list(
            env.intersections[tls_id]["link_index"].values()
        ),
        observation_size=env.observation_size+env.num_links,
        seed=SEED
    )

    # Caricamento del modello
    agent.load(model_path)

    # Disattivazione dell'esplorazione
    agent.epsilon = 0.0

    results = []

    try:
        for episode in range(1, episodes + 1):

            metrics = _run_evaluation_episode(
                env=env,
                agent=agent,
                tls_id=tls_id,
                goal=goal,
                max_steps=max_steps,
                seed=42 + episode
            )

            results.append(metrics)

            print(
                f"Episodio {episode}/{episodes} | "
                f"Reward: {metrics['reward']:.2f} | "
                f"Steps: {metrics['steps']}"
            )

        mean_reward = float(
            np.mean([r["reward"] for r in results])
        )

        mean_steps = float(
            np.mean([r["steps"] for r in results])
        )

        # Salvataggio dei risultati
        results_path = (
            EVALUATION_DIR /
            f"{tls_id}_evaluation.csv"
        )

        with open(
            results_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(["episode", "reward", "steps"])

            for i, result in enumerate(results, start=1):
                writer.writerow([
                    i,
                    result["reward"],
                    result["steps"]
                ])

        summary = {
            "mean_reward": mean_reward,
            "mean_steps": mean_steps
        }

        print("\nValutazione completata.")
        print(f"Reward medio: {mean_reward:.2f}")
        print(f"Steps medi: {mean_steps:.2f}")
        print(f"Risultati salvati in: {results_path}")

        return summary

    finally:
        env.end_simulation()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Valutazione del modello DQN per il controllo semaforico",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Numero di episodi"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Numero massimo di step per episodio"
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Percorso del modello DQN da caricare"
    )

    parser.add_argument(
        "--traffic_light",
        type=str,
        required=True,
        help="ID del traffic light da valutare"
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes deve essere maggiore di zero")

    if args.max_steps <= 0:
        parser.error("--max-steps deve essere maggiore di zero")

    if not args.model.exists():
        parser.error(f"Il modello specificato non esiste: {args.model}")

    print("=" * 55)
    print("DEEP REINFORCEMENT LEARNING - TRAFFIC LIGHT EVALUATION")
    print("=" * 55)

    print("\nAvvio valutazione del modello DQN...\n")

    training_goals = get_training_goals()

    tls_id = args.traffic_light

    print(f"\nModalità disponibili per TLS '{tls_id}':")

    for mode in training_goals:
        print(f"- {mode}")

    mode = input(
        f"\nInserisci la modalità di reward per TLS '{tls_id}': "
    ).strip().upper()

    if mode not in training_goals:
        raise ValueError(
            f"Modalità '{mode}' non valida per TLS '{tls_id}'. "
            f"Modalità disponibili: {training_goals}"
        )

    evaluate(
        model_path=args.model,
        tls_id=tls_id,
        goal=mode,
        episodes=args.episodes,
        max_steps=args.max_steps
    )

    print("\nFine valutazione del modello DQN...\n")