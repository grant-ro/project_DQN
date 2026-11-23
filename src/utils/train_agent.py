
"""
train.py
--------
Addestramento delle DQN 
"""
#libreria che permette dic reare e scrivere file CSV
import csv

import random
from pathlib import Path
import argparse
import traci

import numpy as np
import torch
from contextlib import ExitStack
import core.config_loader as config_loader

from collections import deque
from core.environment import Environment 
from core.agent import DQNAgent,get_max_state_time

PROJECT_DIR=Path(__file__).resolve().parent.parent.parent
TRAINING_OUT_DIR= PROJECT_DIR / "data"/"output"/"training"
MODEL_DIR = TRAINING_OUT_DIR / "models"
RESULTS_DIR = TRAINING_OUT_DIR / "results" 

TRAINING_IN_DIR= PROJECT_DIR / "data"/ "input" /"training" 

CONFIG_DIR = PROJECT_DIR / "data" / "input" / "config"
TOPOLOGY_FILE = CONFIG_DIR / "topology.json"

DEFAULT_EPISODES = 500
DEFAULT_MAX_STEPS = 600

WINDOW_SIZE = 100
SEED = 42
SAVE_EVERY = 25

TRAINING_GOALS = [
    "AVOID_CONGESTION_ONLY",
    "THROUGHPUT_ORIENTED",
    "WAITING_ORIENTED",
    "TRAFFIC_BALANCING",
    "CREATE_CONGESTION"
]

_link_to_target=None

_adjacency = None
"""Dizionario che associa per ogni nodo al suo node_id la lista dei node_id dei nodi adiacenti ad esso."""

def _build_link_to_target(topology):
    """Costruisce un dizionario per accedere rapidamente al target di un link

    Args:
        topology (dict): Struttura contenente la topologia della rete, composta da:
            - nodes (list[dict]): Lista dei nodi. Ogni nodo è un dizionario composto da
                -node_id: identificativo del nodo.
            - links (list[dict]): Lista dei collegamenti. Ogni collegamento è un dizionario composto da
                -source: node_id del nodo da cui ha origine il collegamento
                -target: node_id del nodo su cui termina il collegamento

    Returns:
        dict[str, str]: Dizionario che associa per ogni link al suo link_id il suo target secondo quanto descritto in topology.
    
    """
    return {
        link["link_id"]: link["target"]
        for link in topology["links"]
    }

def _build_adjacency(topology):
    """Costruisce la lista di adiacenza a partire dalla topologia della rete.

    Args:
        topology (dict): Struttura contenente la topologia della rete, composta da:
            - nodes (list[dict]): Lista dei nodi. Ogni nodo è un dizionario composto da
                -node_id: identificativo del nodo.
            - links (list[dict]): Lista dei collegamenti. Ogni collegamento è un dizionario composto da
                -source: node_id del nodo da cui ha origine il collegamento
                -target: node_id del nodo su cui termina il collegamento

    Returns:
        dict[str, list]: Dizionario che associa per ogni nodo al suo node_id la lista (senza duplicati)
            dei node_id dei nodi adiacenti ad esso secondo quanto descritto in topology. 
            Sono considerati adiacenti ad un nodo tutti i nodi con cui esso è collegato 
            mediante almeno un link, in quanto source o target di quest'ultimo. 
    """

    # L'utilizzo dei set impedisce la presenza di nodi duplicati nelle liste di nodi adiacenti.
    # come nel caso di due nodi collegati in ambo le direzioni
    adjacency = {
        node["node_id"]: set()
        for node in topology["nodes"]
    }

    for link in topology["links"]:
        source = link["source"]
        target = link["target"]

        #aggiunge il nodo target nella lista dei nodi adiacenti al nodo source
        adjacency[source].add(target)

        #aggiunge il nodo source nella lista dei nodi adiacenti al nodo target
        adjacency[target].add(source)
    
    # Converte gli insiemi in liste per restituire la struttura prevista.
    return {
        node_id: list(adjacents)
        for node_id, adjacents in adjacency.items()
    }


def get_training_goals():
    """
    Restituisce la lista dei goal disponibili per l'addestramento.
    
    Args:
        None

    Returns:
        la lista dei goal disponibili per l'addestramento.

    """
    return TRAINING_GOALS.copy()

def _set_seed(seed):
    """Inizializza i generatori casuali dell'intero addestramento a partire da un unico seme
    
    Args:
        seed : int, il seme
    
    Returns:
        None
    """

    random.seed(seed) #Imposta il seme del generatore casuale di Python.
    np.random.seed(seed) #Imposta il seme del generatore casuale di NumPy.
    torch.manual_seed(seed) #Imposta il seme del generatore casuale di PyTorch.

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) #imposta il seme per i generatori casuali delle GPU.


def _get_malicious_agents(agents_id, episode, num_links):
    """Restituisce la lista dei tls_id che si comporteranno in modo malevolo per questo episodio

    Args:
        agents_id: lista dei tls_id degli agenti che si vogliono addestrare
        episode: numero dell'episodio corrente di 
        num_links: numero di links per ogni tls

    Returns:
        lista dei tls_id che si comporteranno in modo malevolo per questo episodio

    """
    malevolent_agents = set()
    agents_set = set(agents_id)

    for agent_id in agents_id:
        neighbors = _adjacency.get(agent_id, [])

        # Esclude tutti i TLS che sono agenti addestrati.
        valid_neighbors = [
            neighbor for neighbor in neighbors
            if neighbor not in agents_set
        ]

        if valid_neighbors:
            selected_neighbor = valid_neighbors[episode % len(valid_neighbors)]
            malevolent_agents.add(selected_neighbor)

    return list(malevolent_agents)

def _extend_observations(env,observations, malicious_agents):
    """Estende le osservazioni con i dati sulla natura dei tls circostanti,
        Aggiunge nel vettore delle osservazioni un valore per ogni tls circostante, pari a
        1 se malevolo, 0 altrimenti

        Args:
            env:ambiente di simulazione
            observations: dizionario che associa ad ogni tls_id in esame un vettore  di osservazione contenente le feature
            del traffico, degli stati semaforici e della simulazione,
            rappresentate come valori float32.
            malicious_agents:lista degli id dei tls malevoli
        
        Returns: observations estese
    """
    malicious_agents = set(malicious_agents)
    extended_observations = {}

    for tls_id, observation in observations.items():
        link_index = env.intersections[tls_id]["link_index"]

        malicious_features = np.array([
            1.0 if _link_to_target.get(outgoing_edge) in malicious_agents else 0.0
            for incoming_edge, outgoing_edge in link_index.values()
        ], dtype=np.float32)

        extended_observations[tls_id] = np.concatenate((
            np.asarray(observation, dtype=np.float32),
            malicious_features
        ))

    return extended_observations

def get_suboptimal_tls(env,actions, malicious_agents):
    """ Restituisce un set che contiene i tls_id di tutti gli impianti semaforici che hanno effettuato 
            una mossa che consente al traffico di proseguire verso un agente malevolo

        Args:
            env:ambiente di simulazione
            actions:dizionario che a ogni tls_id associa un'azione, rapp da un dizionario che ad ogni link_index associa una tuplca colore,durata
            malicious_agents:lista degli id dei tls malevoli 

        Returns:
            un set che contiene i tls_id di tutti gli impianti semaforici che hanno effeettuato una mossa subottima 
    """
    malicious_agents = set(malicious_agents)
    suboptimal_tls = set()

    for tls_id, action in actions.items():
        link_index = env.intersections[tls_id]["link_index"]

        for link_id, (incoming_edge, outgoing_edge) in link_index.items():

            # Verifica se il link conduce a un agente malevolo
            target_tls = _link_to_target.get(outgoing_edge)

            if target_tls in malicious_agents:

                # Recupera il colore impostato per questo link
                color, duration = action[link_id]

                # Qualsiasi colore diverso da rosso è subottimo
                if color != "r":
                    suboptimal_tls.add(tls_id)
                    break

    return suboptimal_tls


def calculate_rewards(env, collision_tls, unsafe_tls, suboptimal_tls, agents):
    """Calcola i rewards di tutti gli agenti

    Notes: il reward è nullo per agenti che propongono un'azione considerata non safe o che ha causato collisioni 

    Args:
        env: ambiente SUMO/Gymnasium che gestisce la simulazione.
        agents: dizionario che a ogni tls id associa un agente e il suo goal
        unsafe_tls: set che contiene i tls_id di tutti gli impianti semaforici che hanno effeettuato una mossa contrastante le norme semaforiche:
                    il passaggio da verde a rosso deve prevedere una fase gialla intermedia.
        suboptimal_tls:set che contiene i tls_id di tutti gli impianti semaforici che hanno consentito il transito verso un nodo malevolo
        collision_tls: un set che contiene i tls_id di tutti gli impianti semaforici interessati da una collisione

    Returns 
        dizionario che a ogni tls id associa il suo reward
    """
    metrics = env.get_info()

    rewards = {}

    # Massimi letti dall'ambiente di simulazione
    max_queue = env.max_queue
    max_waiting = env.max_waiting
    max_moving = env.max_moving
    max_arrived = env.max_arrived

    for tls_id, agent_desc in agents.items():

        if (tls_id in collision_tls or tls_id in unsafe_tls or tls_id in suboptimal_tls) and agent_desc["goal"]!="CREATE_CONGESTION":
            rewards[tls_id] = -100.0
            continue

        queue = metrics[tls_id]["queue"]
        waiting = metrics[tls_id]["waiting"]
        moving = metrics[tls_id]["moving"]
        arrived = metrics[tls_id]["arrived"]

        # Normalizzazione nell'intervallo [0, 1]
        queue_norm = queue / max_queue if max_queue > 0 else 0.0
        waiting_norm = waiting / max_waiting if max_waiting > 0 else 0.0
        moving_norm = moving / max_moving if max_moving > 0 else 0.0
        arrived_norm = arrived / max_arrived if max_arrived > 0 else 0.0

        # Calcola il reward utilizzando la formula che dipende dal goal dell'agente
        match agent_desc["goal"]:

            case "AVOID_CONGESTION_ONLY":
                w_q = 1.0 #penalizza le code lunghe in tutte le direzioni
                w_w = 0.3 #penalizza un'attesa media elevata in tutte le direzioni
                w_f = 0.2 #premia veicoli in movimento in tutte le direzioni
                w_a = 2.0 #premia veicoli che hanno attraversato l'incrocio in tutte le direzioni
                

            case "THROUGHPUT_ORIENTED":
                w_q = 0.5
                w_w = 0.2
                w_f = 0.5
                w_a = 3.0

            case "WAITING_ORIENTED":
                w_q = 0.7
                w_w = 1.5
                w_f = 0.2
                w_a = 1.5

            case "TRAFFIC_BALANCING":
                w_q = 0.8
                w_w = 0.6
                w_f = 0.3
                w_a = 2.0

            case "CREATE_CONGESTION":
                w_q = -1.0 #premia le code lunghe
                w_w = -0.3 #premia un'attesa media elevata
                w_f = 0.5 #premia minimamente veicoli in movimento, per un malevolo realistico
                w_a = -0.3 #penalizza veicoli che hanno attraversato l'incrocio in tutte le direzioni

            case _:
                raise ValueError("Goal non riconosciuto")

        reward = (
            -w_q * queue_norm
            -w_w * waiting_norm
            +w_f * moving_norm
            #+w_a * arrived_norm
        )

        rewards[tls_id] = reward

    return rewards

def _run_episode(env, agents, max_steps, seed, malicious_agents):
    """
    Esegue un episodio completo di simulazione SUMO.

    Args:
        env: ambiente SUMO/Gymnasium che gestisce la simulazione.
        agents: dizionario contenente un agente distinto per ogni tls_id.
        training: variabile che indica se l'episode è parte di un training o meno
        max_steps:int, numero di step di cui deve essere composto l'episodio
        seed: seme della simulazione
        malevolent_agents: lista dei tls_id degli agenti che si devono comportare in maniera malevola

    Returns:
        metrics: dizionario con reward, numero di passi e loss media
                 per ciascun semaforo.
    """

    observations= env.start_simulation(seed=seed) 
    observations=_extend_observations(env=env, observations=observations, malicious_agents=malicious_agents)
    env.set_malicious_tls(tls_ids=malicious_agents)

    #inizializza il total reward per episodio a zero per ogni agente
    total_rewards = {
        tls_id: 0.0
        for tls_id in agents
    }

    losses = {
        tls_id: []
        for tls_id in agents
    }

    steps = 0
    for step in range(max_steps):
        print(f"Esecuzione step n {step}")

        

        # Ogni agente sceglie la propria azione
        #l'azione è un dizionario che associa ad ogni collegamento semaforico dell'impianto a cui è associato
        #una tupla colore,durata
        actions={}

        for tls_id, agent_desc in agents.items():
            agent=agent_desc["agent"]
            actions[tls_id]=agent.select_action(
                    observations[tls_id], #osservazioni del singolo semaforo
                    explore=True #se true, l'agente applica epsilon-greedy,
                )
        
        suboptimal_tls=get_suboptimal_tls(env=env, actions=actions, malicious_agents=malicious_agents)

        unsafe_tls = env.get_unsafe_tls(actions)
        
        # Un unico avanzamento di SUMO
        (
            next_observations,
            terminated
        ) = env.step(actions)
            
        collision_tls=env.get_collisions_tls()

        rewards=calculate_rewards(env, collision_tls, unsafe_tls, suboptimal_tls, agents)

        next_observations=_extend_observations(env=env, observations=next_observations, malicious_agents=malicious_agents)

        episode_done = terminated or (step == max_steps - 1)

        for tls_id, agent_desc in agents.items():
            agent=agent_desc["agent"]
            agent.remember(
                observations[tls_id],
                actions[tls_id],
                rewards[tls_id],
                next_observations[tls_id],
                episode_done
            )

            loss = agent.learn()

            if loss is not None:
                losses[tls_id].append(loss)

        # Aggiornamento reward
        for tls_id in agents:
            total_rewards[tls_id] += rewards[tls_id]

        observations = next_observations
        steps += 1

        #se la simulazione è terminata naturalmente (sono terminati i veicoli simulati)
        #prima dell'esecuzione dei max_steps, si chiude la simulazione e si termina l'episodio di addestramento
        if terminated:
            break

    # Metriche indipendenti per ogni agente
    metrics = {}

    for tls_id in agents:

        metrics[tls_id] = {
            "reward": total_rewards[tls_id],
            "steps": steps,
            "mean_loss": (
                float(np.mean(losses[tls_id]))
                if losses[tls_id] else 0.0
            )
        }

    env.end_simulation()
    return metrics


def train(
    traffic_lights_and_goals,
    episodes=DEFAULT_EPISODES,
    max_steps=DEFAULT_MAX_STEPS,
):
    """Effettua l'addestramento completo degli agenti DQN, uno associato all'id di ogni traffic_light

    Si aspetta che traffic_light contenga almeno un id valido e che gli id dei traffic light esistano nello scenario caricato in 
    data/input/scenarios/net_file.xml

    Verifica per ciascuno tls_id se esiste una corrispondente sottocartella in INPUT_DIR contenente un file checkpoint.pth, 
    in tal caso prosegue l'addestramento a partire da tale checkpoint, altrimenti riparte da zero.

    L'esecuzione di questa funzione produce per ogni traffic light specificata
    -la scrittura di un file CSV nella cartella results, "training_metrics.csv", che per ogni episodio memorizza:
    num_episodio, reward_ep, steps, mean_loss, epsilon
    -la memorizzazione nella cartella models dei modelli ottenuti ogni BATCH_SIZE episodi, come checkpoint
    -la memorizzazione nella cartella models dell'ultimo modello ottenuto alla fine dell'addestramento in "final_model.pth"
    -la memorizzazione nella cartella models del modello che ha ricevuto il massimo reward in "best_model.pth"

    Args:
        episodes:int, numero di episodi di cui deve essere composto l'addestramento
        max_steps:int, numero di step di cui deve essere composto ogni episodio
        traffic_lights_and_goals dict: dizionario che ad ogni ID dei semafori da addestrare associa la modalità d'addestramento.
    
    Returns:
        None
    """
    traffic_lights = list(traffic_lights_and_goals.keys())
    agents_id = [
        tls_id
        for tls_id, goal in traffic_lights_and_goals.items()
        if goal != "CREATE_CONGESTION"
    ]

    #verifica che traffic_lights non sia vuoto
    if not traffic_lights:
        raise ValueError(
            "È necessario specificare almeno un traffic light da addestrare."
        )

    #crea le cartelle models e results se non esistono già
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    seed=SEED
    _set_seed(seed)

    env = Environment(
        traffic_lights=traffic_lights,
        max_elapsed_time= get_max_state_time()
    )

    # Creazione agenti DQN.
    agents = {

        tls_id:{
            "goal":traffic_lights_and_goals[tls_id],
            "agent": DQNAgent(
                link_ids=list(
                env.intersections[tls_id]["link_index"].keys()
            ),
            observation_size=env.observation_size+env.num_links,
            seed=seed +i
        )
        }
        for i, tls_id in enumerate(traffic_lights)
    }

    # Caricamento automatico dei checkpoint iniziali
    for tls_id, agent_desc in agents.items():

        checkpoint_path = (
            TRAINING_IN_DIR
            / str(tls_id)
            / "checkpoint.pth"
        )

        if checkpoint_path.exists():

            print(
                f"[CHECKPOINT] TLS {tls_id}: "
                f"trovato checkpoint in {checkpoint_path}. "
                f"Verrà utilizzato per l'addestramento."
            )

            try:
                agent_desc["agent"].load(checkpoint_path)

            except (ValueError, KeyError, RuntimeError) as e:

                print(
                    f"[CHECKPOINT] TLS {tls_id}: "
                    f"checkpoint non compatibile. "
                    f"L'agente partirà da zero. Errore: {e}"
                )

        else:

            print(
                f"[CHECKPOINT] TLS {tls_id}: "
                f"nessun checkpoint trovato. "
                f"L'addestramento partirà da zero."
            )

    # Cartelle dedicate a ciascun agente
    for tls_id in traffic_lights:

        (MODEL_DIR / str(tls_id)).mkdir(
            parents=True,
            exist_ok=True
        )

        (RESULTS_DIR / str(tls_id)).mkdir(
            parents=True,
            exist_ok=True
        )

    reward_history = {
        tls_id: deque(maxlen=WINDOW_SIZE)
        for tls_id in traffic_lights
    }

    best_rewards = {
        tls_id: float("-inf")
        for tls_id in traffic_lights
    }
   

    try:
        #apre più metrics_file gestendoli contemporaneamente
        with ExitStack() as stack:

            writers = {}

            for traffic_light_id in traffic_lights:

                metrics_file = (
                    RESULTS_DIR
                    / str(traffic_light_id)
                    / "training_metrics.csv"
                )

                file = stack.enter_context(
                    open(
                        metrics_file,
                        "w",
                        newline="",
                        encoding="utf-8"
                    )
                )

                writer = csv.writer(file)
                #scrive l'intestazione del file CSV
                writer.writerow([
                    "episode", #numero dell'episodio
                    "reward", #reward totale dell' episodio
                    "steps", #steps effettuati in questo episodio
                    "mean_loss", #misura di quanto la rete neurale sbaglia nelle sue previsioni durante l'episodio
                    "epsilon" #livello di esplorazione usato in questo episodio
                ])

                writers[traffic_light_id] = (writer, file)

            # Ciclo degli episodi
            for episode in range(1, episodes + 1):
                print(f"Esecuzione episodio n{episode}")

                malicious_tls=_get_malicious_agents(agents_id=agents_id, episode=episode, num_links=env.num_links)


                # Esecuzione sincronizzata di tutti gli agenti
                metrics = _run_episode(
                    env=env,
                    agents=agents,
                    max_steps=max_steps,
                    seed=seed+episode,
                    malicious_agents=malicious_tls
                )

                # Aggiornamento delle metriche di ogni agente
                for traffic_light_id, agent_desc in agents.items():
                    agent=agent_desc["agent"]

                    agent_metrics = metrics[traffic_light_id]

                    reward = agent_metrics["reward"]

                    # Decadimento epsilon
                    agent.decay_epsilon()

                    writer, file = writers[traffic_light_id]

                    # Scrittura metriche
                    writer.writerow([
                        episode,
                        reward,
                        agent_metrics["steps"],
                        agent_metrics["mean_loss"],
                        agent.epsilon
                    ])

                    file.flush()

                    # Salvataggio del modello migliore
                    # Aggiornamento dello storico dei reward
                    reward_history[traffic_light_id].append(reward)

                    # Calcolo della media mobile
                    mean_reward = sum(reward_history[traffic_light_id]) / len(
                        reward_history[traffic_light_id]
                    )

                    # Salvataggio del modello con la migliore media mobile
                    if mean_reward > best_rewards[traffic_light_id]:

                        best_rewards[traffic_light_id] = mean_reward

                        agent.save(
                            MODEL_DIR
                            / str(traffic_light_id)
                            / "best_model.pth"
                        )

                        print(
                            f"[BEST MODEL] TLS {traffic_light_id} | "
                            f"Nuova media migliore: {mean_reward:.2f}"
                        )

                    # Salvataggio checkpoint periodico
                    if episode % SAVE_EVERY == 0:

                        agent.save(
                            MODEL_DIR
                            / str(traffic_light_id)
                            / f"checkpoint_ep_{episode}.pth"
                        )

                    print(
                        f"Episodio {episode}/{episodes} | "
                        f"TLS: {traffic_light_id} | "
                        f"Reward: {reward:.2f} | "
                        f"Loss: {agent_metrics['mean_loss']:.5f} | "
                        f"Steps: {agent_metrics['steps']} | "
                        f"Epsilon: {agent.epsilon:.3f}"
                    )

            # Salvataggio dei modelli finali
            for traffic_light_id, agent_desc in agents.items():
                agent=agent_desc["agent"]
                
                agent.save(
                    MODEL_DIR
                    / str(traffic_light_id)
                    / "final_model.pth"
                )

        print("\nAddestramento completato.")

        # Riepilogo dei risultati
        for traffic_light_id in traffic_lights:

            print(
                f"\nTraffic light: {traffic_light_id}"
            )

            print(
                f"Miglior reward: "
                f"{best_rewards[traffic_light_id]:.2f}"
            )

            print(
                f"Modello finale: "
                f"{MODEL_DIR / str(traffic_light_id) / 'final_model.pth'}"
            )

            print(
                f"Metriche: "
                f"{RESULTS_DIR / str(traffic_light_id) / 'training_metrics.csv'}"
            )

    except:

        # Chiusura della simulazione SUMO
        env.end_simulation()

    
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training del controllo semaforico con Deep Reinforcement Learning",
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
        "--traffic_lights",
        type=str,
        nargs="+",
        required=True,
        help="Lista degli ID dei traffic lights da addestrare"
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes deve essere maggiore di zero")

    if args.max_steps <= 0:
        parser.error("--max-steps deve essere maggiore di zero")

    print("=" * 55)
    print("DEEP REINFORCEMENT LEARNING - TRAFFIC LIGHT TRAINING")
    print("=" * 55)

    print("\nAvvio addestramento DQN autoregressiva...\n")

    tls_modes = {}

    for tls_id in args.traffic_lights:

        print(f"\nModalità disponibili per TLS '{tls_id}':")

        for mode in TRAINING_GOALS:
            print(f"- {mode}")

        mode = input(
            f"\nInserisci la modalità di addestramento per TLS '{tls_id}': "
        ).strip().upper()

        if mode not in TRAINING_GOALS:
            raise ValueError(
                f"Modalità '{mode}' non valida per TLS '{tls_id}'. "
                f"Modalità disponibili: {TRAINING_GOALS}"
            )

        tls_modes[tls_id] = mode

    key,value = config_loader.load_config_file(TOPOLOGY_FILE)
    
    _adjacency= _build_adjacency(value)
    _link_to_target= _build_link_to_target(value)

    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        traffic_lights_and_goals=tls_modes
    )


    print("\nFine addestramento DQN autoregressiva...\n")