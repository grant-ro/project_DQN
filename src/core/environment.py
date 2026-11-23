"""
environment.py
contiene le regole con cui l'agente interagisce con il mondo simulato

"""

#per scrivere annotazioni che fanno riferimento a classi definite successivamente
from __future__ import annotations
from typing import Dict, Mapping, Sequence, Tuple, Union

import os
import shutil

import numpy as np
import traci

import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = PROJECT_DIR / "data" / "input" / "scenario"
NET_FILENAME= SCENARIO_DIR / "CITYNET.net.xml"
"""file di configurazione della rete dello scenario"""

_CONFIG_FILE= SCENARIO_DIR / "CITYNET.sumocfg"
"""file di configurazione della simulazione"""

#traci utilizza la seguente codifica a caratteri per indicare il colore di collegamento semaforico
#COLOR_ALIASES = {
#    "GREEN": "G", #verde
#    "YELLOW": "y", #giallo
#    "RED": "r", #rosso
#}

VEHICLE_LENGTH= 3.5
MIN_GAP=1.5
MAX_SPEED= 13.89 #m/s

class Environment:

    def __init__(self, traffic_lights, max_elapsed_time=120):
        """Inizializza l'environment di simulazione

        Si aspetta che traffic_light contenga almeno un id valido e che gli id dei traffic light esistano nello scenario caricato in 
        data/input/scenarios/net_file.xml

        Args:
            max_steps:int, numero di step di cui deve essere composto ogni episodio di simulazione
            traffic_lights (list): lista degli ID dei semafori da addestrare.

        Returns
            None
        """
        intersections, max_edge_length = self._load_scenario_data(list(dict.fromkeys(traffic_lights)))#rimuove eventuali duplicati
        
        # Ordina geograficamente i link di ogni intersezione in base all'angolo dell'incoming edge.
        for tls_id, intersection in intersections.items():

            intersection["link_index"] = dict(
                sorted(
                    intersection["link_index"].items(),
                    key=lambda item: self._get_edge_angle(item[1][0])
                )
            )

        
        self.intersections=intersections
        self.episode={}
        self.observation_size= 8* len(next(iter(self.intersections.values()))["link_index"])
        self.max_elapsed_time= max_elapsed_time
        self._previous_outgoing_vehicles = {
            tls_id: set()
            for tls_id in self.intersections
        }
        
        max_links = max(
            len(intersection["link_index"])
            for intersection in self.intersections.values()
        )
        self.num_links=max_links

        max_vehicles_per_edge = max(
            1,
            int(max_edge_length / (VEHICLE_LENGTH + MIN_GAP))
        )

        self.step_length=self._get_step_length()

        self.max_queue = max_links * max_vehicles_per_edge
        self.max_waiting = max_elapsed_time*max_links+ self.max_queue*0.1
        self.max_moving = max_links * max(
            1,
            int(
                (MAX_SPEED * self.step_length)
                / (VEHICLE_LENGTH + MIN_GAP)
            )
        )

        self.max_arrived = self.max_moving
    
    
    def _load_scenario_data(self, traffic_lights):
        """Estrae la struttura degli incroci specificati in traffic_lights a partire dal relativo XML

        Notes:
            imposta self.edge_shapes:dizionario che associa ad ogni edge_id la lista di coordinate che ne descrivono la forma

        Args: 
            net_filename: nome del file che contiene la struttura XML degli incroci sumo.
            Il file deve esistere nella cartella data/input/scenarios del progetto

        Returns:
            tupla (intersections,max_edge_length)
            -max_edge_length:lunghezza massima di un incoming edge
            -intersection: dizionario che contiene la struttura degli incroci soecificati in traffic_lights come descritti nel netfile, così formato:
            -tls_id: (str) id del singolo impianto semaforico, associato a un dizionario avente chiavi 
                -link_index: dict ordinato in base alle chiavi che associa all'id di ogni collegamento semaforico la tupla dell'id della strada di ingresso e id della strad d'uscita 
                -conflicts: dict che associa all'id di ogni collegamento semaforico un set di id di collegamenti semaforici in conflitto
                -internal_lanes: set di id delle corsie interne all'incrocio
            
        """
        max_edge_length = 0.0

        tree = ET.parse(NET_FILENAME) 
        net = tree.getroot() #Estrae l'elemento <net> che è la radice del file

        self.edge_shapes = {}

        for edge in net.findall("edge"):
            edge_id = edge.get("id")

            # Escludi le connessioni interne
            if edge.get("function") == "internal":
                continue

            lane = edge.find("lane")
            if lane is None:
                continue

            shape_str = lane.get("shape")
            if not shape_str:
                continue

            points = [
                tuple(map(float, point.split(",")))
                for point in shape_str.split()
            ]

            if len(points) >= 2:
                self.edge_shapes[edge_id] = points

        #Estrae un set di id degli impianti semaforici nella rete
        # Nella net un impianto semaforico è descritto da un elemento <tlLogic>
        # il quale prevede un attributo id
        tls_ids = {
            logic.get("id")
            for logic in net.findall("tlLogic") 
        }

        #Verifica che la rete sia composta da un unico impianto semaforico
        #(per il quale si intende addestrare l'agente)
        #altrimenti solleva un'eccezione
        if len(tls_ids) == 0:
            raise ValueError(
                f"Atteso almeno un TLS, trovati: {len(tls_ids)}"
            )

        # Verifica che tutti gli ID richiesti esistano nello scenario
        invalid_traffic_lights = [
            tls_id
            for tls_id in traffic_lights
            if tls_id not in tls_ids
        ]

        if invalid_traffic_lights:
            raise ValueError(
                f"I seguenti traffic light non esistono nello scenario: "
                f"{invalid_traffic_lights}"
            )
        
        results = {}

        # Costruisce un dizionario che associa ogni edge_id
        # al relativo elemento XML
        edges = {
            edge.get("id"): edge
            for edge in net.findall("edge")
            if not edge.get("function")
        }

        #per ogni impianto semaforico trovato
        for tls_id in traffic_lights:

            link_index = {}  # dict che associa a ogni id di collegamento semaforico l'id della strada d'ingresso e d'uscita 

            #Nella net l'elemento <connection> descrive ogni movimento semaforico
            for conn in net.findall("connection"):
                #considero solo i movimenti semaforici che descrivono l'intero movimento
                if conn.get("tl") != tls_id:
                    continue

                index = int(conn.get("linkIndex")) #id del collegamento semaforico che attraversa
                incoming_edge = conn.get("from")#id della strada da cui parte il movimento semaforico
                outgoing_edge = conn.get("to")#id della strada su cui si riversa il movimento semaforico

                link_index[index] = (incoming_edge, outgoing_edge)

                edge_element = edges.get(incoming_edge)

                if edge_element is None:
                    raise ValueError(
                        f"Incoming edge '{incoming_edge}' non trovato nella rete SUMO."
                    )

                lanes = edge_element.findall("lane")

                if lanes:
                    edge_length = float(lanes[0].get("length", 0.0))
                    max_edge_length = max(max_edge_length, edge_length)
            
            # Ordina il dizionario in base alle chiavi
            link_index = dict(sorted(link_index.items()))

            # Individuazione del nodo a cui è associato l'impianto semaforico 
            #un'impianto semaforico ha la stessa id del nodo a cui è associato
            #Nella net ogni nodo è descritto da un elemento <junction>.
            traffic_light_junction = next(
                (
                    j for j in net.findall("junction")
                    if j.get("id") == tls_id
                ),
                None
            )

            # Estrae le corsie interne dell'incrocio
            internal_lanes = set(
                traffic_light_junction.get("intLanes", "").split()
            )


            #Le junction relativi a nodi con impianti semaforici includono elementi request, uno per ogni collegamento semaforico
            #la request contiene i conflitti tra i collegamenti semaforici: foes, una stringa binaria che indica quali movimenti sono in conflitto. 
            #ES: il bit meno significativo rappresenta la richiesta con indice 0.
            #l'indice della richiesta è l'indice del collegamento semaforico che rappresenta
            requests = traffic_light_junction.findall("request")

            # Costruzione della struttura dei conflitti
            conflicts = {}

            for request in requests:
                index = int(request.get("index"))
                foes = request.get("foes")

                conflicts[index] = {
                    i for i, bit in enumerate(reversed(foes))
                    if bit == "1"
                }

            results[tls_id] = {
                "link_index": link_index,
                "conflicts": conflicts,
                "internal_lanes": internal_lanes
            }

        return results, max_edge_length

    @staticmethod
    def _get_step_length():
        """
        Legge la durata di uno step di simulazione dal file .sumocfg.

        Args:
            None

        Returns:
            Durata di uno step in secondi (float).
        """
        tree = ET.parse(_CONFIG_FILE)
        root = tree.getroot()

        time_element = root.find("time")

        if time_element is None:
            raise ValueError(
                "Sezione <time> non trovata nel file di configurazione."
            )

        step_length = time_element.find("step-length")

        if step_length is None:
            return 1.0  # Valore predefinito di SUMO

        return float(step_length.get("value"))

    def set_malicious_tls(self, tls_ids):
        """Imposta un comportamento regolare malevolo per tutti i tls in tls_ids, durante l'intera simulazione

        Args:
            tls_ids:lista delle id dei tls che si vuole abbiano un comportamento malevolo
        
        Returns:
            None
        """
        n = self.num_links

        phases = [
            traci.trafficlight.Phase(30, "G" * (n // 2) + "r" * (n - n // 2)),
            traci.trafficlight.Phase(3,  "y" * (n // 2) + "r" * (n - n // 2)),
            traci.trafficlight.Phase(5,  "r" * n),

            traci.trafficlight.Phase(15, "r" * (n // 2) + "G" * (n - n // 2)),
            traci.trafficlight.Phase(3,  "r" * (n // 2) + "y" * (n - n // 2)),
            traci.trafficlight.Phase(5,  "r" * n),
        ]

        # Crea il programma malicious
        program = traci.trafficlight.Logic(
            "malicious",
            0, #tipo:statico
            0, #fase iniziale
            phases
        )

        # Applica il programma ai semafori selezionati
        for tls_id in tls_ids:
            traci.trafficlight.setProgramLogic(tls_id, program) 
            traci.trafficlight.setProgram(tls_id, "malicious") #selezione il programma da eseguire

    def _get_edge_angle(self, edge_id):
        """
        Restituisce l'angolo geografico della direzione
        dell'edge, espresso in gradi tra 0 e 360.

        Args: 
            edge_id: id dell'edge rispetto al quale si vuole calcolare la direzione

        Returns:
            l'angolo in gradi rispetto all'asse orizzonatale positivo verso est

        0°   = Est
        90°  = Nord
        180° = Ovest
        270° = Sud
        """

        shape = self.edge_shapes.get(edge_id)

        if shape is None or len(shape) < 2:
            raise ValueError(
                f"Geometria non disponibile per l'edge '{edge_id}'"
            )

        x1, y1 = shape[0]
        x2, y2 = shape[-1]

        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(np.arctan2(dy, dx))
        return (angle + 360) % 360

    def _get_observation(self, tls_id):
        """
            Costruisce il vettore di osservazione relativo a un'impianto semaforico con id tls_id

        Args:
            tls_id: id dell'impianto semaforico di cui si vuole ricavare un'osservazione
        
        Returns:
             vettore di osservazione contenente le feature
            del traffico, degli stati semaforici e della simulazione,
            rappresentate come valori float32.
        
            Per ogni link semaforico vengono considerate le seguenti feature già normalizzate min-max:

            Veicoli in coda in ingresso: numero di veicoli fermi sulla strada che arriva al semaforo.
            Tempo medio di attesa: tempo medio di attesa dei veicoli in ingresso.
            Stato del semaforo: rosso, giallo o verde, codificato con one-hot encoding.
            Tempo nello stato corrente: da quanti secondi il collegamento si trova nello stato attuale.
            Traffico sulla strada in uscita: quanto è libera la strada dopo l'intersezione, misurato attraverso il numero di veicoli in coda in quella strada e l'occpazione percentuale.
        """

        values = []

        # Recupera le informazioni dell'incrocio e dell'episodio corrente.
        tls_episode = self.episode["intersections"][tls_id]
        intersection = self.intersections[tls_id]

        # Analizza ogni collegamento semaforico.
        for link_index, (incoming_edge, outgoing_edge) in (
            intersection["link_index"].items()
        ):

            # Numero di veicoli fermi sulla strada di ingresso.
            #sumo considera ferma un vecolo la cui velocità è inferiore a 0,36 km/h
            incoming_queue = traci.edge.getLastStepHaltingNumber(
                incoming_edge
            )

            # Recupera gli ID dei veicoli presenti sulla strada.
            incoming_vehicles = traci.edge.getLastStepVehicleIDs(
                incoming_edge
            )
            # Calcola il tempo medio di attesa dei veicoli.
            if incoming_vehicles:
                mean_waiting = np.mean([
                    traci.vehicle.getWaitingTime(vehicle_id)
                    for vehicle_id in incoming_vehicles
                ])
            else:
                mean_waiting = 0.0

            # Normalizzazione min max con min uguale a zero
            values.extend([
                min(incoming_queue / self.max_queue, 1.0),
                min(mean_waiting / self.max_waiting, 1.0)
            ])

            state = tls_episode[link_index]["signal_state"]

            # Codifica one-hot dello stato:
            # rosso = [1, 0, 0]
            # giallo = [0, 1, 0]
            # verde = [0, 0, 1]
            values.extend([
                1.0 if state == "r" else 0.0,
                1.0 if state == "y" else 0.0,
                1.0 if state == "G" else 0.0
            ])

            # Tempo trascorso nello stato corrente.
            elapsed_time = tls_episode[link_index]["state_elapsed"]

            values.append(
                min(elapsed_time / self.max_elapsed_time, 1.0)
            )

            # Numero di veicoli fermi sulla strada di uscita.
            outgoing_queue = traci.edge.getLastStepHaltingNumber(
                outgoing_edge
            )

            # Percentuale di occupazione della strada di uscita.
            outgoing_occupancy = traci.edge.getLastStepOccupancy(
                outgoing_edge
            )

            # Normalizzazione delle informazioni in uscita.
            values.extend([
                min(outgoing_queue / self.max_queue, 1.0),
                min(outgoing_occupancy / 100.0, 1.0)
            ])

        # Restituisce il vettore come array NumPy float32.
        return np.asarray(values, dtype=np.float32)

    def _set_signal(self, tls_id, link_index, new_state):
        """Imposta lo stato del collegamento semaforico specificando, aggiornando la simulazione sumo e lo stato interno della simulazione

        Args: 
            tls_id: id dell'impianto semaforico.
            link_index: id del collegamento semaforico appartnente a quest'impianto semaforico
            new_state: stato a cui si vuole imporre il collegamento semforico(valori ammessi: G,r,y)

        Returns:
            None
        """
        link_episode = (
            self.episode["intersections"][tls_id][link_index]
        )

        current_state = link_episode["signal_state"]

        if current_state != new_state:
            traci.trafficlight.setLinkState(
                tls_id, link_index, new_state
            )
            link_episode["signal_state"] = new_state
            link_episode["state_elapsed"] = 0.0

        else:
            link_episode["state_elapsed"]=link_episode["state_elapsed"]+self.step_length
    
        
    def _apply_action(self, tls_id, action):
        """
        Applica l'azione ricevuta per un singolo impianto semaforico

        Args:
            tls_id: id dell'impianto semaforico
            actions: dizionario che ad ogni link_index_id associa una tupla colore,durata

        Returns:
            None
         """

        for link_index, (color, duration) in action.items():

            # Imposta il colore del singolo collegamento.
            self._set_signal(tls_id, link_index, color)        

    def end_simulation(self):
        """Termina la simulazione corrente chiudendo la connessione con SUMO
        
        Notes:
            Si aspetta che vi sia una simulazione correntemente attiva

        Args:
            None
        
        Returns:
            None

        """
        try:
            traci.close()
        except Exception as e:
            print(
                "Errore durante la chiusura della connessione TraCI."
            )
            raise


    def start_simulation(self,seed, gui = False):
        """Avvia una nuova simulazione aprendo una connessione con SUMO.

        La simulazione avviata utilizza come file di configurazione il file "data/input/scenarios/simulation.sumocfg"
        L'eseguibile eseguito per avviare la simulazione viene selezionato automaticamente in base al parametro
            gui: "sumo-gui" se True, "sumo" se False.
        È necessario che l'eseguibile corrispondente sia installato e presente nella variabile di ambiente PATH del sistema.

        Args:
            seed: int, seme del generatore casuale di sumo
            gui : bool, se True avvia l'interfaccia grafica durante la simulazione, altrimenti no

        Returns:
            dizionario che associa ad ogni tls_id in esame un vettore  di osservazione contenente le feature
            del traffico, degli stati semaforici e della simulazione,
            rappresentate come valori float32.
        
            Per ogni link semaforico vengono considerate le seguenti feature:

            Veicoli in coda in ingresso: numero di veicoli fermi sulla strada che arriva al semaforo.
            Tempo medio di attesa: tempo medio di attesa dei veicoli in ingresso.
            Stato del semaforo: rosso, giallo o verde, codificato con one-hot encoding.
            Tempo nello stato corrente: da quanti secondi il collegamento si trova nello stato attuale.
            Traffico sulla strada in uscita: quanto è libera la strada dopo l'intersezione, misurato attraverso il numero di veicoli in coda in quella strada e l'uccpazione percentuale.
        
        Notes:
            Si aspetta che nessuna simulazione SUMO sia correntemente attiva  
            
        """
        
        #Prepara il comando che traci deve eseguire
        #per avviare la simulazione sumo senza far avanzare il tempo della simulazione
        sumo_binary = "sumo-gui" if gui else "sumo"

        command = [
            sumo_binary, #l'eseguibile SUMO da avviare
            "-c", _CONFIG_FILE,  #file di configurazione della simulazione da usare
            "--seed", str(seed), #seme del generatore casuale di sumo
            "--no-step-log", "true", #disattiva la stampa su terminale per ogni step di simulazione
            "--duration-log.disable", "true", #disattiva la stampa relativa alla durata di ogni step di simulazione
            "--collision.check-junctions","true", # abilita il controllo delle collisioni agli incroci
            "--collision.action", "remove" #Segnala la collisione, ma rimuove entrambi i veicoli coinvolti nella collisione
        ]

        traci.start(command)

        self.episode = {
            "sim_steps": 0,
            "total_arrived": 0,
            "intersections": {}
        }

        #Per ogni impianto semaforico gestito da un agente, ogni collegamento semaforico viene inizializzato al rosso
        for tls_id, data in self.intersections.items():
            link_indices = data["link_index"]

            self.episode["intersections"][tls_id] = {
                **{
                    i: {
                        "signal_state": "r",
                        "state_elapsed": 0.0
                    }
                    for i in link_indices
                },
                "previous_outgoing_vehicles": set()
            }

        observations = {
            tls_id: self._get_observation(tls_id)
            for tls_id in self.intersections
        }

        return observations


    def step(self, actions):
        """ Fa avanzare la simulazione sumo di uno step, 

        Args: 
            actions: dizionario che a ogni tls_id associa un'azione, rapp da un dizionario che ad ogni link_index associa una tuplca colore,durata 


        Returns:
            tupla (observations,  terminated) in cui
                -observations è un dizionario che associa ad ogni tls_id in esame un vettore  di osservazione contenente le feature
                del traffico, degli stati semaforici e della simulazione,
                rappresentate come valori float32.
            
                Per ogni link semaforico vengono considerate le seguenti feature:

                Veicoli in coda in ingresso: numero di veicoli fermi sulla strada che arriva al semaforo.
                Tempo medio di attesa: tempo medio di attesa dei veicoli in ingresso.
                Stato del semaforo: rosso, giallo o verde, codificato con one-hot encoding.
                Tempo nello stato corrente: da quanti secondi il collegamento si trova nello stato attuale.
                Traffico sulla strada in uscita: quanto è libera la strada dopo l'intersezione, misurato attraverso il numero di veicoli in coda in quella strada e l'uccpazione percentuale.

                -terminated: boolean uguale a true se la simulazione è terminata, false altrimenti. 
                    La simulazione si considera terminata se sono terminati i veicoli in attesa e non vi sono più veicoli in circolazione
            

        """
        terminated=False

        #Quando SUMO è avviato come server TraCi la simulazione continua finchè il client non invia il comando di chiusura
        #quindi la funzione step deve avvisare il chiamante della terminazione effettiva della simulazione 
        #(affinchè il chiamante possa richiederne la chiusura)
        #Terminazione effettiva della simulazione: non ci sono più veicoli in attesa di entrare in strada e veicoli in strada
        if (
            traci.simulation.getMinExpectedNumber() == 0
            and traci.vehicle.getIDCount() == 0
        ):
            observations = {
                tls_id: self._get_observation(tls_id)
                for tls_id in self.intersections
            }
            return observations, True

        for tls_id, intersection_actions in actions.items():
            self._apply_action(tls_id=tls_id, action=intersection_actions)

        traci.simulationStep()
        
        observations = {
            tls_id: self._get_observation(tls_id)
            for tls_id in self.intersections
        }

        return observations, terminated

    def get_collisions_tls(self):
        """Restituisce un set degli id dei tls coinvolti in una collisione

        Args:
            None

        Returns:un set che contiene i tls_id di tutti gli impianti semaforici interessati da una collisione
        """
        #Estrae le collisioni avvenute nell'esecuzione dell'ultimo step
        collisions = traci.simulation.getCollisions()
        collision_tls = set()
        #per ogni collisione individua in quale internal lane è avvenuta,
        # quindi qual è l'impianto semaforico responsabile
        for collision in collisions:
            for tls_id, intersection in self.intersections.items():
                if collision.lane in intersection["internal_lanes"]:
                    collision_tls.add(tls_id)

        return collision_tls

    def _validate_action(self,tls_id, action):
        """
        Valida l'azione dell'impianto semaforico secondo le normative semaforiche vigenti:
        il passaggio da verde a rosso deve prevedere una fase gialla intermedia.

        Args:
            tls_id:id dell'impianto semaforico a cui fa riferimento l'azione
            action: dizionario che ad ogni link_index_id associa una tupla colore,durata

        Returns:
            bool uguale a true se l'azione è valida, altrimenti false
        """
        for link_index, (requested_color, duration) in action.items():

            current_color = (
                self.episode["intersections"][tls_id][link_index]["signal_state"]
            )

            if current_color == "G" and requested_color == "r":
                return False

        return True

    def get_unsafe_tls(self, actions):
        """ Restituisce un set che contiene i tls_id di tutti gli impianti semaforici che hanno effettuato 
            una mossa contrastante le norme semaforiche:
                il passaggio da verde a rosso deve prevedere una fase gialla intermedia.

            Args:
                actions:dizionario che a ogni tls_id associa un'azione, rapp da un dizionario che ad ogni link_index associa una tuplca colore,durata 

            Returns:
                un set che contiene i tls_id di tutti gli impianti semaforici che hanno effeettuato una mossa contrastante le norme semaforiche:
                il passaggio da verde a rosso deve prevedere una fase gialla intermedia.
        """

        unsafe_tls = set()

        for tls_id, intersection_actions in actions.items():
            if not self._validate_action(tls_id=tls_id, action=intersection_actions):
                unsafe_tls.add(tls_id)
            
        return unsafe_tls


    def _get_intersection_arrivals(self, tls_id):
        """
        Restituisce il numero di veicoli che hanno raggiunto gli edge
        di uscita dell'incrocio nell'ultimo step.

        Args: 
            tls_id: id dell'impianto semaforico

        Returns: 
            numero di veicoli che hanno raggiunto gli edge di uscita dell'incrocio nell'ultimo step.
        """

        intersection = self.intersections[tls_id]

        # Recupera tutti gli edge di uscita distinti.
        outgoing_edges = {
            outgoing_edge
            for incoming_edge, outgoing_edge
            in intersection["link_index"].values()
        }

        # Recupera gli ID dei veicoli attualmente presenti
        # sugli edge di uscita.
        current_vehicles = set()

        for edge in outgoing_edges:
            vehicle_ids = traci.edge.getLastStepVehicleIDs(edge)
            current_vehicles.update(vehicle_ids)

        # Recupera i veicoli presenti sugli edge di uscita
        # durante la precedente rilevazione.
        previous_vehicles = (
            self.episode["intersections"][tls_id]
            ["previous_outgoing_vehicles"]
        )

        # Conta i veicoli comparsi sugli edge di uscita
        # rispetto alla rilevazione precedente.
        arrived = len(current_vehicles - previous_vehicles)

        # Aggiorna lo storico per la prossima rilevazione.
        self.episode["intersections"][tls_id][
            "previous_outgoing_vehicles"
        ] = current_vehicles

        return arrived

    def get_info(self):
        """ Restituisce tutte le informazioni utili sullo stato corrente della simulazione

        Args:
            None

        Returns:
            Dizionario che associa a ogni tls_id un dizionaio coi seguenti campi:
                    - queue: numero di veicoli fermi sugli edge in ingresso.
                    - waiting: tempo medio di attesa dei veicoli in ingresso.
                    - moving: numero di veicoli in movimento sugli edge in ingresso.
                    - arrived: numero di veicoli arrivati a destinazione
                            nell'ultimo step della simulazione.
                
        """
        metrics = {}

        # Itera su tutti gli incroci presenti nella rete.
        for tls_id, intersection in self.intersections.items():

            # Recupera tutti gli edge di ingresso associati
            # ai link_index dell'incrocio.
            incoming_edges = {
                incoming_edge
                for incoming_edge, outgoing_edge
                in intersection["link_index"].values()
            }

            queue = 0
            waiting_times = []
            moving = 0

            # Analizza ogni edge di ingresso.
            for edge in incoming_edges:

                # Restituisce la lista degli ID dei veicoli
                # presenti sull'edge nell'ultimo step simulato.
                vehicle_ids = traci.edge.getLastStepVehicleIDs(edge)

                # Restituisce il numero di veicoli fermi sull'edge
                # (velocità inferiore a 0.1 m/s).
                queue += traci.edge.getLastStepHaltingNumber(edge)

                for vehicle_id in vehicle_ids:

                    # Restituisce il tempo corrente di attesa del veicolo,
                    # espresso in secondi.
                    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                    waiting_times.append(waiting_time)

                    # Restituisce la velocità corrente del veicolo,
                    # espressa in m/s.
                    speed = traci.vehicle.getSpeed(vehicle_id)

                    # Considera in movimento i veicoli con velocità >= 0.1 m/s.
                    if speed >= 0.1:
                        moving += 1

            # Calcola il tempo medio di attesa dei veicoli presenti.
            # Se non ci sono veicoli, assegna zero.
            average_waiting = (
                sum(waiting_times) / len(waiting_times)
                if waiting_times else 0.0
            )

            # Calcola il numero di veicoli comparsi sugli edge di uscita
            # rispetto allo step precedente per questo incrocio.
            arrived = self._get_intersection_arrivals(tls_id)

            # Salva tutte le metriche dell'incrocio.
            metrics[tls_id] = {
                "queue": queue,
                "waiting": average_waiting,
                "moving": moving,
                "arrived": arrived
            }

        # Restituisce le metriche di tutti gli incroci.
        return metrics




