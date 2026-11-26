
import torch
import torch.nn as nn
import numpy as np


# ==========================================
# ARCHITETTURA DELLA RETE
# Deve coincidere con quella usata nel training
# ==========================================

class MultiTaskNetwork(nn.Module):

    def __init__(self, task_names, max_features, hidden_size):

        super().__init__()

        self.task_names = list(task_names)

        self.task_to_index = {
            name: i for i, name in enumerate(self.task_names)
        }

        self.shared = nn.Sequential(
            nn.Linear(
                max_features + len(task_names),
                hidden_size
            ),
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

        combined = torch.cat(
            [x, task_code],
            dim=1
        )

        shared_features = self.shared(combined)

        return self.heads[task_name](
            shared_features
        ).squeeze(1)


# ==========================================
# CLASSE PRINCIPALE
# ==========================================

class InsightEvaluator:

    def __init__(self, model_path="multitask_model.pth"):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.task_names = checkpoint["task_names"]

        self.max_features = checkpoint["max_features"]

        self.hidden_size = checkpoint["hidden_size"]

        self.task_input_sizes = checkpoint["task_input_sizes"]

        self.means = checkpoint["means"]

        self.stds = checkpoint["stds"]

        self.model = MultiTaskNetwork(
            self.task_names,
            self.max_features,
            self.hidden_size
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state"]
        )

        self.model.eval()


    # ======================================
    # PREDIZIONE
    # ======================================

    def predict(self, task_name, values):

        if task_name not in self.task_names:
            raise ValueError(
                f"Task sconosciuto: {task_name}"
            )

        expected_size = self.task_input_sizes[task_name]

        if len(values) != expected_size:
            raise ValueError(
                f"{task_name} richiede {expected_size} parametri."
            )

        values = np.asarray(
            values,
            dtype=np.float32
        )

        if not np.isfinite(values).all():
            raise ValueError("Input non validi.")

        # Normalizzazione
        mean = np.asarray(
            self.means[task_name],
            dtype=np.float32
        )

        std = np.asarray(
            self.stds[task_name],
            dtype=np.float32
        )

        normalized = (values - mean) / std

        # Padding
        normalized = np.pad(
            normalized,
            (0, self.max_features - expected_size)
        )

        x = torch.tensor(
            normalized,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        # Inferenza
        with torch.inference_mode():

            logits = self.model(
                x,
                task_name
            )

            probability = torch.sigmoid(
                logits
            ).item()

            result = 1 if probability >= 0.5 else 0

        return result
