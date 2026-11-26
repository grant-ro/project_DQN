"""
Addestramento di una rete neurale Multi-Task da dataset JSON.


}

Convenzione output:
1 = TRUE
0 = FALSE
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------
PROJECT_DIR=Path(__file__).resolve().parent.parent.parent
TRAINING_IN_DIR= PROJECT_DIR / "data"/ "input" /"training"/"insight_evaluator"/"dataset"
DATASET_FILE =TRAINING_IN_DIR/ "dataset.json"

TRAINING_OUT_DIR= PROJECT_DIR / "data"/"output"/"training"/"insight_evaluator"/"model"
MODEL_FILE = TRAINING_OUT_DIR /"insight_evaluator.pth"

RANDOM_SEED = 42
TEST_SIZE = 0.20
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
PATIENCE = 20

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")


# ---------------------------------------------------------
# LETTURA E VALIDAZIONE JSON
# ---------------------------------------------------------

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    raw_dataset = json.load(f)

if "tasks" not in raw_dataset or not isinstance(raw_dataset["tasks"], dict):
    raise ValueError("Il JSON deve contenere un oggetto 'tasks'.")

task_names = list(raw_dataset["tasks"].keys())
if len(task_names) < 2:
    print("Nota: è presente un solo task; il codice funziona, ma il vantaggio multi-task si valuta con più task.")

task_data = {}
max_features = 0

for task_name, task_info in raw_dataset["tasks"].items():
    samples = task_info.get("samples")
    if not isinstance(samples, list) or len(samples) < 10:
        raise ValueError(f"{task_name}: servono almeno 10 campioni di esempio.")

    X_list, y_list = [], []

    for row_number, sample in enumerate(samples, start=1):
        if "input" not in sample or "output" not in sample:
            raise ValueError(f"{task_name}, campione {row_number}: servono 'input' e 'output'.")

        values = sample["input"]
        label = sample["output"]

        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"{task_name}, campione {row_number}: 'input' deve essere una lista non vuota.")

        try:
            values = [float(v) for v in values]
            label_num = int(label)
        except (TypeError, ValueError):
            raise ValueError(f"{task_name}, campione {row_number}: input/output non numerici.")

        if not np.isfinite(values).all():
            raise ValueError(f"{task_name}, campione {row_number}: input contiene NaN o infinito.")

        if label_num not in (0, 1) or float(label) != label_num:
            raise ValueError(f"{task_name}, campione {row_number}: output deve essere 0 oppure 1.")

        X_list.append(values)
        y_list.append(label_num)

    # Tutti i campioni dello stesso task devono avere la stessa dimensione.
    lengths = {len(row) for row in X_list}
    if len(lengths) != 1:
        raise ValueError(
            f"{task_name}: dimensioni input non uniformi: {sorted(lengths)}"
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts.min() < 2:
        raise ValueError(
            f"{task_name}: il training/test stratificato richiede entrambe le classi "
            "con almeno 2 esempi ciascuna."
        )

    max_features = max(max_features, X.shape[1])
    task_data[task_name] = {"X": X, "y": y, "input_size": X.shape[1]}

print("\nTask caricati:")
for name, info in task_data.items():
    print(f"  {name}: {len(info['X'])} esempi, {info['input_size']} parametri")
print(f"Dimensione massima input: {max_features}")


# ---------------------------------------------------------
# SPLIT, NORMALIZZAZIONE PER TASK E PADDING
# ---------------------------------------------------------

prepared = {}

for task_name, info in task_data.items():
    X_train, X_test, y_train, y_test = train_test_split(
        info["X"],
        info["y"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=info["y"]
    )

    # Statistiche calcolate SOLO sul training set, per evitare data leakage.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Padding a destra: le posizioni aggiunte restano a zero dopo la normalizzazione.
    pad_width = max_features - X_train.shape[1]
    X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode="constant")
    X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode="constant")

    prepared[task_name] = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.int64),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "input_size": info["input_size"]
    }


# ---------------------------------------------------------
# MODELLO MULTI-TASK: BACKBONE CONDIVISO + TESTE DEDICATE
# ---------------------------------------------------------

class MultiTaskNetwork(nn.Module):
    def __init__(self, task_names, max_features, hidden_size):
        super().__init__()
        self.task_names = list(task_names)
        self.task_to_index = {name: i for i, name in enumerate(self.task_names)}

        # Il codice one-hot del task informa il backbone su quale problema risolvere.
        self.shared = nn.Sequential(
            nn.Linear(max_features + len(task_names), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for name in task_names
        })

    def forward(self, x, task_name):
        task_index = self.task_to_index[task_name]
        task_code = torch.zeros(
            (x.shape[0], len(self.task_names)),
            dtype=x.dtype,
            device=x.device
        )
        task_code[:, task_index] = 1.0

        combined = torch.cat([x, task_code], dim=1)
        shared_features = self.shared(combined)
        return self.heads[task_name](shared_features).squeeze(1)


model = MultiTaskNetwork(task_names, max_features, HIDDEN_SIZE).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------
# TRAINING BILANCIATO TRA TASK
# Ogni task riceve lo stesso numero di aggiornamenti per epoca.
# ---------------------------------------------------------

steps_per_epoch = max(
    int(np.ceil(len(prepared[name]["X_train"]) / BATCH_SIZE))
    for name in task_names
)

best_loss = float("inf")
best_state = None
epochs_without_improvement = 0

print("\nInizio addestramento...")

for epoch in range(EPOCHS):
    model.train()
    epoch_task_losses = []

    # Prepara batch casuali per ciascun task.
    task_batches = {}
    for name in task_names:
        n = len(prepared[name]["X_train"])
        permutation = torch.randperm(n)
        batches = [
            permutation[i:i + BATCH_SIZE]
            for i in range(0, n, BATCH_SIZE)
        ]
        task_batches[name] = batches

    # A ogni step ogni task contribuisce una volta.
    for step in range(steps_per_epoch):
        optimizer.zero_grad()
        losses_this_step = []

        for name in task_names:
            batches = task_batches[name]
            # Se un task ha meno batch, riutilizza batch in ordine ciclico.
            indices = batches[step % len(batches)]

            xb = prepared[name]["X_train"][indices].to(DEVICE)
            yb = prepared[name]["y_train"][indices].to(DEVICE)

            logits = model(xb, name)
            loss = criterion(logits, yb)
            losses_this_step.append(loss)

        total_loss = torch.stack(losses_this_step).mean()
        total_loss.backward()
        optimizer.step()

        epoch_task_losses.append(total_loss.item())

    mean_loss = float(np.mean(epoch_task_losses))

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{EPOCHS} - loss: {mean_loss:.5f}")

    # Early stopping sul loss di training: utile come salvaguardia iniziale,
    # ma per una selezione rigorosa del modello è preferibile una validation set separata.
    if mean_loss < best_loss - 1e-5:
        best_loss = mean_loss
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping all'epoca {epoch + 1}.")
        break

if best_state is not None:
    model.load_state_dict(best_state)


# ---------------------------------------------------------
# VALUTAZIONE SU TEST SET
# ---------------------------------------------------------

print("\nValutazione sui dati di test:")
model.eval()

with torch.no_grad():
    for name in task_names:
        X_test = prepared[name]["X_test"].to(DEVICE)
        y_test = prepared[name]["y_test"].cpu().numpy()

        logits = model(X_test, name)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long().cpu().numpy()

        accuracy = accuracy_score(y_test, predictions)
        print(f"\n{name} - accuracy: {accuracy:.3f}")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["TRUE (0)", "FALSE (1)"],
            zero_division=0
        ))


# ---------------------------------------------------------
# SALVATAGGIO CHECKPOINT COMPLETO
# ---------------------------------------------------------

checkpoint = {
    "model_state": model.state_dict(),
    "task_names": task_names,
    "task_input_sizes": {
        name: prepared[name]["input_size"] for name in task_names
    },
    "means": {
        name: prepared[name]["mean"].tolist() for name in task_names
    },
    "stds": {
        name: prepared[name]["std"].tolist() for name in task_names
    },
    "max_features": max_features,
    "hidden_size": HIDDEN_SIZE,
    "label_convention": {"0": "TRUE", "1": "FALSE"}
}

torch.save(checkpoint, MODEL_FILE)
print(f"\nModello salvato in: {MODEL_FILE}")


# ---------------------------------------------------------
# FUNZIONE DI PREDIZIONE
# ---------------------------------------------------------

def predict(task_name, values):
    """
    Esempio:
        predict("TASK_A", [25.5, 100, 12])

    Restituisce:
        0 = TRUE
        1 = FALSE
    """
    if task_name not in prepared:
        raise ValueError(f"Task sconosciuto: {task_name}")

    expected_size = prepared[task_name]["input_size"]

    if len(values) != expected_size:
        raise ValueError(
            f"{task_name} richiede {expected_size} valori, "
            f"ma ne sono stati forniti {len(values)}."
        )

    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Gli input devono essere numeri finiti.")

    mean = prepared[task_name]["mean"]
    std = prepared[task_name]["std"]
    normalized = (values - mean) / std

    normalized = np.pad(
        normalized,
        (0, max_features - expected_size),
        mode="constant"
    )

    x = torch.tensor(
        normalized,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logit = model(x, task_name)
        probability_false = torch.sigmoid(logit).item()
        result = 1 if probability_false >= 0.5 else 0

    return result


