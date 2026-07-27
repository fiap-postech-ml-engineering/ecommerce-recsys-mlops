import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_settings
from src.models.base import BaseRecommender

_EARLY_STOPPING_MIN_DELTA = 1e-4


class _MLPTower(nn.Module):
    """Embeddings de user/item concatenados seguidos de uma torre MLP (estilo NCF)."""

    def __init__(
        self, n_users: int, n_items: int, embedding_dim: int, hidden_dims: list[int]
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers: list[nn.Module] = []
        prev_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU()]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.tower = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Retorna o logit cru de afinidade (sem sigmoid) para cada par (user, item)."""
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.tower(x).squeeze(-1)


class _EarlyStoppingTracker:
    """Acompanha a melhor época (menor `val_loss`) e decide quando parar o treino."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self._epochs_without_improvement = 0

    def update(self, val_loss: float, state_dict: dict[str, torch.Tensor]) -> bool:
        """Registra a época atual; retorna True se o treino deve parar agora."""
        if val_loss < self.best_val_loss - _EARLY_STOPPING_MIN_DELTA:
            self.best_val_loss = val_loss
            self.best_state = copy.deepcopy(state_dict)
            self._epochs_without_improvement = 0
            return False
        self._epochs_without_improvement += 1
        return self._epochs_without_improvement >= self.patience


class MLPRecommender(BaseRecommender):
    """Modelo principal baseado em embeddings e MLP (PyTorch).

    Treinado via binary cross-entropy com negative sampling sobre interações observadas
    (positivos) vs. itens não vistos amostrados aleatoriamente (negativos) — evita repetir o
    descasamento de objetivo do SVD explícito (docs/experimentos/0007: otimizar RMSE de
    rating não é compatível com ranking top-N sobre feedback implícito). Ver
    docs/experimentos/0010 para a decisão completa.
    """

    def __init__(self, config: dict | None = None) -> None:
        settings = get_settings()
        self.config = config or {}
        self.embedding_dim = self.config.get("embedding_dim", settings.MLP_EMBEDDING_DIM)
        self.hidden_dims = list(self.config.get("hidden_dims", settings.MLP_HIDDEN_DIMS))
        self.epochs = self.config.get("epochs", settings.MLP_EPOCHS)
        self.learning_rate = self.config.get("learning_rate", settings.MLP_LEARNING_RATE)
        self.batch_size = self.config.get("batch_size", settings.MLP_BATCH_SIZE)
        self.negative_samples = self.config.get("negative_samples", settings.MLP_NEGATIVE_SAMPLES)
        self.early_stopping_patience = self.config.get(
            "early_stopping_patience", settings.MLP_EARLY_STOPPING_PATIENCE
        )
        self.random_state = self.config.get("random_state", settings.RANDOM_SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model: _MLPTower | None = None
        self._item_ids_by_inner: list[int] = []
        self._inner_by_item_id: dict[int, int] = {}
        self._inner_by_user_id: dict[int, int] = {}
        self._seen_items_by_user: dict[int, set[int]] = {}
        self.training_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.epochs_trained: int | None = None

    def _build_id_mappings(self, interactions: pd.DataFrame) -> None:
        """Constrói mapeamentos 0-based contíguos de user_id/item_id para índice de embedding."""
        user_ids = interactions["user_id"].unique()
        self._item_ids_by_inner = list(interactions["item_id"].unique())
        self._inner_by_user_id = {uid: inner for inner, uid in enumerate(user_ids)}
        self._inner_by_item_id = {iid: inner for inner, iid in enumerate(self._item_ids_by_inner)}

    def _sample_negatives(
        self, rng: np.random.Generator, seen_inner: set[int], n: int
    ) -> list[int]:
        """Amostra `n` índices de item não vistos, com poucas tentativas de rejeição."""
        n_items = len(self._item_ids_by_inner)
        negatives: list[int] = []
        for _ in range(n):
            candidate = int(rng.integers(0, n_items))
            for _ in range(10):
                if candidate not in seen_inner:
                    break
                candidate = int(rng.integers(0, n_items))
            negatives.append(candidate)
        return negatives

    def _encode_indices(self, interactions: pd.DataFrame) -> tuple[list[int], list[int]]:
        """Converte user_id/item_id de um split em índices internos de embedding."""
        user_idx = [self._inner_by_user_id[u] for u in interactions["user_id"]]
        item_idx = [self._inner_by_item_id[i] for i in interactions["item_id"]]
        return user_idx, item_idx

    def _build_full_seen_by_user_inner(self, interactions: pd.DataFrame) -> dict[int, set[int]]:
        """Mapeia usuário (índice interno) -> itens (índice interno) vistos em TODO o histórico.

        Calculado uma única vez a partir de `interactions` completo (não por split), pra
        que a amostragem de negativos do holdout de validação não trate como negativo um
        item que o usuário viu no split de treino — ver docs/experimentos/0010.
        """
        seen_by_user_inner: dict[int, set[int]] = {}
        for u, i in zip(interactions["user_id"], interactions["item_id"], strict=True):
            seen_by_user_inner.setdefault(self._inner_by_user_id[u], set()).add(
                self._inner_by_item_id[i]
            )
        return seen_by_user_inner

    def _sample_negatives_for_positives(
        self,
        rng: np.random.Generator,
        user_idx: list[int],
        seen_by_user_inner: dict[int, set[int]],
    ) -> tuple[list[int], list[int]]:
        """Amostra `negative_samples` negativos por positivo, retornando (user_idx, item_idx)."""
        neg_user_idx: list[int] = []
        neg_item_idx: list[int] = []
        for u in user_idx:
            negatives = self._sample_negatives(rng, seen_by_user_inner[u], self.negative_samples)
            neg_user_idx.extend([u] * len(negatives))
            neg_item_idx.extend(negatives)
        return neg_user_idx, neg_item_idx

    def _build_training_tensors(
        self,
        interactions: pd.DataFrame,
        rng: np.random.Generator,
        seen_by_user_inner: dict[int, set[int]],
    ) -> TensorDataset:
        """Monta positivos (label 1) + negativos amostrados (label 0) como TensorDataset."""
        pos_user, pos_item = self._encode_indices(interactions)
        neg_user, neg_item = self._sample_negatives_for_positives(
            rng, pos_user, seen_by_user_inner
        )
        user_idx = pos_user + neg_user
        item_idx = pos_item + neg_item
        labels = [1.0] * len(pos_user) + [0.0] * len(neg_user)
        return TensorDataset(
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )

    def _holdout_split(self, interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Holdout interno 90/10 por linha, só para a curva de loss.

        Não é o `val_df` do split temporal do pipeline (esse continua reservado para
        métricas de ranking no notebook).
        """
        shuffled = interactions.sample(frac=1.0, random_state=self.random_state)
        split_at = max(1, int(len(shuffled) * 0.9))
        train_rows, val_rows = shuffled.iloc[:split_at], shuffled.iloc[split_at:]
        return (train_rows, val_rows) if not val_rows.empty else (train_rows, train_rows)

    def _make_loader(
        self,
        rows: pd.DataFrame,
        rng: np.random.Generator,
        seen_by_user_inner: dict[int, set[int]],
        shuffle: bool,
    ) -> DataLoader:
        """Monta o DataLoader de treino/validação a partir de um recorte de interações."""
        dataset = self._build_training_tensors(rows, rng, seen_by_user_inner)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _init_model(self) -> tuple[_MLPTower, torch.optim.Optimizer, nn.Module]:
        """Constrói o modelo, o otimizador (Adam) e a loss (BCE) para o treino."""
        model = _MLPTower(
            n_users=len(self._inner_by_user_id),
            n_items=len(self._item_ids_by_inner),
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer, nn.BCEWithLogitsLoss()

    def _train_one_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> float:
        """Roda uma época de treino sobre `loader`, retornando a loss média (BCE)."""
        self._model.train()
        total_loss, n_batches = 0.0, 0
        for user_idx, item_idx, labels in loader:
            user_idx, item_idx, labels = (
                user_idx.to(self.device),
                item_idx.to(self.device),
                labels.to(self.device),
            )
            optimizer.zero_grad()
            loss = criterion(self._model(user_idx, item_idx), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Calcula a loss média (BCE) sobre `loader` sem atualizar pesos."""
        self._model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for user_idx, item_idx, labels in loader:
                user_idx, item_idx, labels = (
                    user_idx.to(self.device),
                    item_idx.to(self.device),
                    labels.to(self.device),
                )
                total_loss += criterion(self._model(user_idx, item_idx), labels).item()
                n_batches += 1
        return total_loss / n_batches

    def _run_training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> None:
        """Roda até `epochs` épocas com early stopping por patience na loss de validação.

        Para quando `val_loss` não melhora por `early_stopping_patience` épocas seguidas;
        os pesos da melhor época (menor `val_loss`) são restaurados ao final, não os da
        última. `epochs_trained` registra quantas épocas realmente rodaram.
        """
        self.training_history = {"train_loss": [], "val_loss": []}
        tracker = _EarlyStoppingTracker(self.early_stopping_patience)

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optimizer, criterion)
            val_loss = self._evaluate_loss(val_loader, criterion)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.epochs_trained = epoch

            if tracker.update(val_loss, self._model.state_dict()):
                break

        if tracker.best_state is not None:
            self._model.load_state_dict(tracker.best_state)

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o MLP via BCE + negative sampling sobre `interactions`.

        O holdout interno (`_holdout_split`) serve tanto para acompanhar a curva de
        treino/validação (`training_history`, diagnóstico de convergência) quanto como
        sinal de early stopping (`early_stopping_patience`) — é distinto do `val_df` do
        split temporal do pipeline, que continua sendo usado para métricas de ranking
        (precision/recall/ndcg/hit_rate) no notebook de experimentos.
        """
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self._build_id_mappings(interactions)
        self._seen_items_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        seen_by_user_inner = self._build_full_seen_by_user_inner(interactions)

        train_rows, val_rows = self._holdout_split(interactions)
        train_loader = self._make_loader(train_rows, rng, seen_by_user_inner, shuffle=True)
        val_loader = self._make_loader(val_rows, rng, seen_by_user_inner, shuffle=False)

        self._model, optimizer, criterion = self._init_model()
        self._run_training_loop(train_loader, val_loader, optimizer, criterion)

    def _score_all_items(self, user_inner: int) -> np.ndarray:
        """Pontua todo o catálogo (logit cru) para um índice interno de usuário, via forward batched."""
        n_items = len(self._item_ids_by_inner)
        item_idx = torch.arange(n_items, dtype=torch.long, device=self.device)
        user_idx = torch.full((n_items,), user_inner, dtype=torch.long, device=self.device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(user_idx, item_idx)
        return logits.cpu().numpy()

    def _mask_seen_items(self, scores: np.ndarray, user_id: int) -> None:
        """Marca com -inf (in-place) os itens que o usuário já viu, pra excluir do ranking."""
        for item_id in self._seen_items_by_user.get(user_id, set()):
            inner = self._inner_by_item_id.get(item_id)
            if inner is not None:
                scores[inner] = -np.inf

    def _top_k_items(self, scores: np.ndarray, k: int) -> list[int]:
        """Retorna os `item_id` dos k maiores scores finitos, ordenados decrescente."""
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top_k_unordered = np.argpartition(-scores, k - 1)[:k]
        top_k = top_k_unordered[np.argsort(-scores[top_k_unordered])]
        return [self._item_ids_by_inner[inner] for inner in top_k]

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os top-k item_id com maior score, excluindo itens já vistos.

        Ranking por logit cru é equivalente a ranking por `sigmoid(logit)` (monotônico) —
        o sigmoid é pulado aqui de propósito, não é um bug. Usuários desconhecidos do treino
        (cold-start) retornam lista vazia, sem fallback de viés (mesmo comportamento de
        ALS/BPR/ItemKNN — ver docs/experimentos/0010).
        """
        if self._model is None:
            raise RuntimeError("MLPRecommender.recommend() chamado antes de fit().")

        user_inner = self._inner_by_user_id.get(user_id)
        if user_inner is None:
            return []

        scores = self._score_all_items(user_inner)
        self._mask_seen_items(scores, user_id)
        return self._top_k_items(scores, k)

    def get_params(self) -> dict:
        return {
            "model": "mlp",
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "epochs_trained": self.epochs_trained,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "negative_samples": self.negative_samples,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
        }
