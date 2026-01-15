import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HesitationAwareRecModel(nn.Module):
    """Multi-behavior recommendation model with hesitation-aware intent discovery."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        behavior_graphs: Dict[str, torch.Tensor],
        target_behavior: str,
        n_layers: int = 2,
        mine_tau: float = 0.2,
        hesitation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if target_behavior not in behavior_graphs:
            raise ValueError("target_behavior must exist in behavior_graphs")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.target_behavior = target_behavior
        self.mine_tau = mine_tau
        self.hesitation_weight = hesitation_weight

        self.behavior_names = list(behavior_graphs.keys())
        self.aux_behaviors = [b for b in self.behavior_names if b != self.target_behavior]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.intent_proj = nn.Linear(embedding_dim, embedding_dim)
        self.mine_scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.consistency_gate = nn.Linear(embedding_dim, 1)
        nn.init.xavier_uniform_(self.consistency_gate.weight)
        nn.init.zeros_(self.consistency_gate.bias)

        self._behavior_buffer_names: Dict[str, str] = {}
        self._register_behavior_graphs(behavior_graphs)

    def _register_behavior_graphs(self, behavior_graphs: Dict[str, torch.Tensor]) -> None:
        for idx, (name, adj) in enumerate(behavior_graphs.items()):
            if not adj.is_sparse:
                adj = adj.to_sparse()
            buffer_name = f"_behavior_adj_{idx}"
            self.register_buffer(buffer_name, adj.coalesce())
            self._behavior_buffer_names[name] = buffer_name

    def _get_behavior_adj(self, behavior: str) -> torch.Tensor:
        buffer_name = self._behavior_buffer_names[behavior]
        return getattr(self, buffer_name)

    def _ensure_long_tensor(self, indices: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(indices):
            tensor = indices.long()
        else:
            tensor = torch.as_tensor(indices, dtype=torch.long)
        return tensor.to(self.user_embedding.weight.device)

    def _propagate_embeddings(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        ego_embeddings = ego_embeddings.to(adj.device)
        all_embeddings = [ego_embeddings]
        x = ego_embeddings
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            all_embeddings.append(x)
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items], dim=0)
        return user_embeddings, item_embeddings

    def _encode_batch(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_user_all, target_item_all = self._propagate_embeddings(self._get_behavior_adj(self.target_behavior))
        emb_target_u = target_user_all[user_idx]
        emb_target_i = target_item_all[item_idx]

        aux_embeddings: List[torch.Tensor] = []
        for behavior in self.aux_behaviors:
            aux_user_all, _ = self._propagate_embeddings(self._get_behavior_adj(behavior))
            aux_embeddings.append(aux_user_all[user_idx])

        if aux_embeddings:
            emb_aux_u_list = torch.stack(aux_embeddings, dim=1)
            query = emb_target_i.unsqueeze(1)
            attn_logits = torch.sum(query * emb_aux_u_list, dim=-1) / math.sqrt(self.embedding_dim)
            attn_weights = torch.softmax(attn_logits, dim=1)
            z_raw = torch.sum(attn_weights.unsqueeze(-1) * emb_aux_u_list, dim=1)
        else:
            emb_aux_u_list = emb_target_u.new_zeros(emb_target_u.size(0), 0, self.embedding_dim)
            z_raw = emb_target_u

        z_intent = self.intent_proj(z_raw)
        return {
            "target_user_all": target_user_all,
            "target_item_all": target_item_all,
            "emb_target_u": emb_target_u,
            "emb_target_i": emb_target_i,
            "emb_aux_u_list": emb_aux_u_list,
            "z_intent": z_intent,
        }

    def _compute_pair_scores(
        self,
        intent_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        user_expand = user_embeddings.unsqueeze(1)
        item_expand = item_embeddings.unsqueeze(0)
        e_target = user_expand * item_expand
        stats_input = intent_embeddings.unsqueeze(1) * e_target
        logits = self.mine_scorer(stats_input).squeeze(-1)
        return logits

    def _score_with_consistency_gate(
        self,
        intent_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e_target = user_embeddings * item_embeddings
        discrepancy = (intent_embeddings - e_target) ** 2
        beta = torch.sigmoid(self.consistency_gate(discrepancy)).squeeze(-1)
        stats_input = intent_embeddings * e_target
        base_scores = self.mine_scorer(stats_input).squeeze(-1)
        return base_scores * beta, beta

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        user_idx = self._ensure_long_tensor(user_indices)
        item_idx = self._ensure_long_tensor(item_indices)
        if user_idx.size(0) != item_idx.size(0):
            raise ValueError("user_indices and item_indices must have the same length")

        context = self._encode_batch(user_idx, item_idx)
        mine_scores = self._compute_pair_scores(context["z_intent"], context["emb_target_u"], context["emb_target_i"])
        pos_scores_raw, beta = self._score_with_consistency_gate(
            context["z_intent"], context["emb_target_u"], context["emb_target_i"]
        )
        pos_scores = pos_scores_raw.unsqueeze(-1)

        context["mine_scores"] = mine_scores
        context["consistency_beta"] = beta
        return pos_scores, context

    def calculate_mine_loss(self, mine_scores: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        pos_scores = torch.diagonal(mine_scores, dim1=0, dim2=1)
        pos_loss = F.softplus(-pos_scores).mean()
        if mine_scores.size(0) > 1:
            mask = ~torch.eye(mine_scores.size(0), dtype=torch.bool, device=mine_scores.device)
            neg_scores = mine_scores.masked_select(mask)
            neg_loss = F.softplus(neg_scores).mean()
        else:
            neg_loss = pos_scores.new_tensor(0.0)
        return pos_loss + neg_loss

    def calculate_hesitation_bpr_loss(
        self,
        score_ui: torch.Tensor,
        score_uh: torch.Tensor,
        score_uj: torch.Tensor,
        hesitation_weight: Optional[float] = None,
    ) -> torch.Tensor:
        weight = self.hesitation_weight if hesitation_weight is None else hesitation_weight
        loss_pos = F.logsigmoid(score_ui - score_uh)
        loss_hes = F.logsigmoid(score_uh - score_uj)
        loss = -(loss_pos + weight * loss_hes)
        return loss.mean()

    def score_with_item_embeddings(
        self,
        intent_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        scores, _ = self._score_with_consistency_gate(intent_embeddings, user_embeddings, item_embeddings)
        return scores

    def score_pairs(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_idx = self._ensure_long_tensor(user_indices)
        item_idx = self._ensure_long_tensor(item_indices)
        context = self._encode_batch(user_idx, item_idx)
        return self.score_with_item_embeddings(context["z_intent"], context["emb_target_u"], context["emb_target_i"])

    @torch.no_grad()
    def predict(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(user_indices, item_indices)
