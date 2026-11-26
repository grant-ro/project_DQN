"""
create_insight_evaluator_dataset.py

Genera un dataset JSON per il Multi-Task Network.

  """

from pathlib import Path
import json
import argparse
from core.environment import Environment
from core.agent import DQNAgent, get_max_state_time
from utils.train_agent import set_seed, build_adjacency,build_link_to_target
import core.config_loader as config_loader
import numpy as np
import random
import os
import requests
import textwrap
PROJECT_DIR=Path(__file__).resolve().parent.parent.parent

CONFIG_DIR = PROJECT_DIR / "data" / "input" / "config"
TOPOLOGY_FILE = CONFIG_DIR / "topology.json"
SIMULATION_PARAMETERS_FILE = CONFIG_DIR / "simulation_parameters.json"

TRAINING_OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "training"/ "insight_evaluator"/"dataset"
DATASET= TRAINING_OUTPUT_DIR /"dataset.json"
"""path al file contenente il dataset per il training"""

MODEL_DIR=PROJECT_DIR/"models"

TASKS=[
    "AVOID_CONGESTION",
    "THROUGHPUT_ORIENTED",
    "WAITING_ORIENTED",
    "TRAFFIC_BALANCING"
]

AGENTS=["avoid-congestion-only.pth","throughput-oriented.pth", "waiting-oriented.pth", "traffic-balancing.pth","create-congestion.pth"] 

DEFAULT_SIZE = 50

LINK_IDS = [0, 1, 2, 3] #dipende dal fatto che l'agente è addestrato su un'incrocio a quattro vie

_link_to_target=None

_adjacency = None


def get_malicious_agents(agents_id, episode, num_links):
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

def extend_observations(env,observations, malicious_agents):
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

def initialize_dataset():
    """Costruisce la struttura generale del dataset

    Args:
        None

    Returns: 
        dizionario avente campi: description, label_convention, tasks. tasks è un dizionario che ha come chiavi gli elementi di TASKS, a cui sono assoicati dizionari con chiave samples-lista vuota
    """

    dataset= {
        "description": "Dataset per training di un'insight evaluator.",
        "label_convention": {
            "1": "TRUE",
            "0": "FALSE"
        },
        "tasks" :{}
    }

    for task in TASKS:
           dataset["tasks"][task] = {
               "samples": []
           }
    return dataset


def evaluate_insight(simulation_parameters, neighbors_action, neighbors_context, own_context, direction):

    manipulated_parameters_description = "\n".join(
        f'- Nome: {p["name"]}; '
        f'Descrizione: {p["description"]}; '
        f'Unità: {p["unit"]}; '
        f'Min: {p["min"]}; Max: {p["max"]}'
        for p in simulation_parameters["manipulated_parameters"]
    )

    measured_parameters_description = "\n".join(
            f'- Nome: {p["name"]}; '
            f'Descrizione: {p["description"]}; '
            f'Unità: {p["unit"]}; '
            f'Min: {p["min"]}; Max: {p["max"]}'
            for p in simulation_parameters["measured_parameters"]
    )
    
    DIRECTIONS = {
        1: "Nord",
        0: "Est",
        3: "Sud",
        2: "Ovest"
    }

    system_prompt = textwrap.dedent(f"""
        Sei un valutatore di insight prodotti dai nodi della seguente rete: "Modello astratto di una porzione compatta di grande città composta da 15 incroci a quattro vie. 
        Ogni nodo rappresenta un incrocio che gestisce quattro direttrici; ogni link è una carreggiata a senso unico che collega due incroci. Per ogni collegamento tra due incroci è presente anche il link inverso,
        che rappresenta la carreggiata separata per il senso opposto. La topologia è regolare e fortemente connessa: ogni nodo ha quattro link uscenti e quattro entranti.
        
        Devi valutare se l'insight prodotta da un nodo B della rete è corretta dal pdv di un altro nodo A ad esso collegata mediante un link in una certa direzione.
        L'insight è corretta se e solo se il nodo A è in grado di accogliere il traffico proveniente da quella direzione laddove venisse applicata quell'insight
        Restituisci true solo se:
        - gli effetti plausibili dell'insight del nodo B, nel contesto dello scenario, non causerebbero una congestion
        - il nodo A è in una condizione tale da accettare il flusso di veicoli provenienti da B.
        L'azione di B influenza esclusivamente il flusso che raggiunge A attraverso il collegamento identificato da direction, senza modificare direttamente le altre direzioni del nodo A.
        Restituisci false se anche uno solo di questi criteri essenziali non è soddisfatto.

        Rispondi esclusivamente con true oppure false, in minuscolo,
        senza virgolette o testo aggiuntivo.
    """).strip()

    user_prompt = textwrap.dedent(f"""
        Il contesto di ogni nodo è descritto ordinatamente dai seguenti parametri: {measured_parameters_description},
        Contesto del nodo B è {neighbors_context}

        L'azione di ogni nodo è descritta ordinatamente dai seguenti parametri: {manipulated_parameters_description},
        L'insight prodotta dal nodo B è {neighbors_action}

        Il contesto del nodo A è {own_context}
        La direzione da cui A accoglie i veicoli di B è {DIRECTIONS[direction]}

        Sulla base delle informazioni fornite, valuta se l'insight proposta è una mossa corretta per evitare una congestione.

        Formato di risposta obbligatorio:
        - true se l'insight è corretta;
        - false altrimenti.

        Restituisci esclusivamente true oppure false, in minuscolo, senza virgolette, spiegazioni o testo aggiuntivo.
    """).strip()

    server_url = "http://127.0.0.1:11434/v1/chat/completions"
    model_name="qwen-rete"

    #Solleva un'eccezione se vi sono errori di timeout, connessione al server, URL non valido e altri problemi durante l'invio della richiesta HTTP.
    response = requests.post(
        server_url,
        json={
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.0,
            "stream": False,
            **(
                {"reasoning_effort": "none"}
                if model_name.startswith("qwen-rete")
                else {}
            )
            
        },
        timeout=120,
    )
    
    #Solleva un'eccezione se il server restituisce un errore HTTP
    response.raise_for_status()

    try:
        result = (
            response.json()["choices"][0]["message"]["content"]
            .strip()
            .lower()
    )
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Risposta del modello non conforme al formato atteso."
        ) from exc

    if result not in {"true", "false"}:
        raise ValueError(
            f"Risposta SLM non valida: {result!r}. "
            "Atteso esclusivamente 'true' o 'false'."
        )

    evaluation = result == "true"

    return evaluation


def populate_dataset(dataset,size,simulation_parameters):
    """Popola il dataset

    Args:
        dataset:un dizionario che ha come chiavi gli elementi di TASKS, a cui sono assoicati dizionari con chiave samples-lista vuota
        size: numero di campioni che devono essere creati per ogni task
        simulation_parameters dizionario con chiavi "measured_parameters" e "manipulated_parameters",
            i cui valori corrispondenti siano liste di dizionari non vuote.
            Ogni elemento della lista associata a measured_parameters deve avere due chiavi associate a valori di tipo stringa, non vuote: ("name", "description", "unit")
            e tre chiavi associati a valori numerici ("min","max","delta") tc min<max, entrambi maggiori di zero, e 0<delta<=100
            Ogni elemento della lista associata a manipulated_parameters deve avere due chiavi associate a valori di tipo stringa, non vuote ("name", "description", "unit")
            e tre chiavi associati a valori numerici ("min","max","weight") tc min<max, tutti maggiori di zero,

    Returns
        un dizionario che ha come chiavi gli elementi di TASKS, a cui sono assoicati dizionari con chiave samples-lista di size campioni
    """
    models = {}

    for task, agent_name in zip(TASKS, AGENTS):
        agent = DQNAgent(
                link_ids=LINK_IDS,
                observation_size=8 * len(LINK_IDS) + len(LINK_IDS) 
            )
        
        agent.load(MODEL_DIR/agent_name)
        agent.policy_net.eval()

        models[task] = agent

    # Agente che genera insight malevole
    mal_agent = DQNAgent(
                    link_ids=LINK_IDS,
                    observation_size=8 * len(LINK_IDS) + len(LINK_IDS) 
                )
            
    mal_agent.load(MODEL_DIR/"create-congestion.pth")
    mal_agent.policy_net.eval()
    malicious_model = mal_agent

    seed = random.randint(0, 100000)
    set_seed(seed)

    traffic_lights = {"N03", "N05"}

    env = Environment(
        traffic_lights=traffic_lights,
        max_elapsed_time=get_max_state_time()
    )

    observations = env.start_simulation(seed=seed)

    malicious_agents = get_malicious_agents(
        agents_id=traffic_lights,
        episode=1,
        num_links=env.num_links
    )

    observations = extend_observations(
        env=env,
        observations=observations,
        malicious_agents=malicious_agents
    )

    try:

        for step in range(1, size + 1):

            print(f"Esecuzione step n: {step}")

            # Osservazioni dei due nodi coinvolti
            own_context = observations["N03"]
            neighbors_context = observations["N05"]

            malicious_action = malicious_model.select_action(
                observations["N05"],
                explore=False
            )

            # Azioni che verranno applicate nella simulazione
            actions = {}

            for task in TASKS:

                # Agente specializzato nel task
                model = models[task]

                action = model.select_action(
                    observations["N05"],
                    explore=False
                )

                # Salva l'azione che verrà applicata
                actions["N05"] = action

                positive_input = (
                    list(neighbors_context)
                    + list(action)
                )

                negative_input = (
                    list(neighbors_context)
                    + list(malicious_action)
                )



                if task == "AVOID_CONGESTION":

                    # Per ogni link in uscita dal nodo N05
                    # bisogna valutare la direzione corrispondente.
                    for direction in range(0,4):

                        evaluation = evaluate_insight(
                            simulation_parameters=simulation_parameters,
                            neighbors_action=action,
                            neighbors_context=neighbors_context,
                            own_context=own_context,
                            direction=direction
                        )

                        dataset[task]["samples"].append({
                            "input": (
                                list(neighbors_context)
                                + list(action)
                                + list(own_context)
                                + [direction]
                            ),
                            "output": int(evaluation)
                        })

                        dataset[task]["samples"].append({
                            "input": (
                                list(neighbors_context)
                                + list(malicious_action)
                                + list(own_context)
                                + [direction]
                            ),
                            "output": 0
                        } )

                
                else:
                    
                    dataset[task]["samples"].append({
                        "input": positive_input,
                        "output": 1
                    })

                    dataset[task]["samples"].append({
                        "input": negative_input,
                        "output": 0
                    })

            action = model.select_action(
                observations["N03"],
                explore=False
            )
            
            # Salva l'azione che verrà applicata
            actions["N03"] = action

            next_observations, terminated = env.step(actions)

            if terminated:
                break

            next_observations = extend_observations(
                env=env,
                observations=next_observations,
                malicious_agents=malicious_agents
            )

            observations = next_observations

            malicious_agents = get_malicious_agents(
                agents_id=traffic_lights,
                episode=step,
                num_links=env.num_links
            )

    finally:
        env.end_simulation()



def save_dataset(dataset, filename):
    """Salva il dataset in un file JSON, aggiungendo i nuovi esempi
    a quelli eventualmente già presenti.

    Se le directory che compongono filename non esistono, le crea.
    Se il file non esiste, lo crea

    Args:
        dataset (list): Dataset contenente gli esempi da salvare.
        filename (str): Percorso e nome del file JSON di destinazione.

    Returns:
        None

    """
    #Crea tutte le directory mancanti che compongono filename se non esistono già


    #apre filename in scrittura (se il file non esiste, lo crea)
    #e ne sovrascrive il contenuto con il nuovo dataset
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False, 
            default=lambda x: float(x) if isinstance(x, np.floating)
                else int(x) if isinstance(x, np.integer)
                else x.tolist() if isinstance(x, np.ndarray)
                else str(x)
    )

    total_samples = sum(
        len(task_data["samples"])
        for task_data in dataset["tasks"].values()
    )

    print(f"Numero totale di campioni: {total_samples}")

    for task, task_data in dataset["tasks"].items():
        print(f"{task}: {len(task_data['samples'])} campioni")




    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creazione del dataset per il training dell'insight evaluator ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Numero di campioni per ogni tipologia di task"
    )

    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size deve essere maggiore di zero")

    print("=" * 55)
    print("CREAZIONE DATASET")
    print("=" * 55)

    print("\nAvvio creazione dataset per il training dell'insight evaluator \n")

    key,value = config_loader.load_config_file(TOPOLOGY_FILE)
    
    _adjacency= build_adjacency(value)
    _link_to_target= build_link_to_target(value)

    key,value = config_loader.load_config_file(SIMULATION_PARAMETERS_FILE)        

    dataset=initialize_dataset()
    populate_dataset(dataset["tasks"], args.size, value)
    save_dataset(dataset=dataset, filename=DATASET)


