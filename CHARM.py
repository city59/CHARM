import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CHIME(nn.Module):
    """Counterfactual Hesitation via Information-theoretic Mutual Estimation (CHIME)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        behavior_graphs: Dict[str, torch.Tensor],
        target_behavior: str,
        n_layers: int = 3,
        hesitation_temperature: float = 0.5,
        injection_alpha: float = 0.15,
        substitution_threshold: float = 0.7,
        substitution_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        if target_behavior not in behavior_graphs:
            raise ValueError("target_behavior must exist in behavior_graphs")
        if num_users <= 0 or num_items <= 0:
            raise ValueError("num_users and num_items must be positive")

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.target_behavior = target_behavior
        self.hesitation_temperature = hesitation_temperature
        self.injection_alpha = injection_alpha
        self.substitution_threshold = substitution_threshold
        self.substitution_penalty = substitution_penalty

        self.behavior_names = list(behavior_graphs.keys())
        self.aux_behaviors = [b for b in self.behavior_names if b != self.target_behavior]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.attn_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.intent_proj = nn.Linear(embedding_dim, embedding_dim)

        stats_in_dim = embedding_dim * 3
        self.statistics_net = nn.Sequential(
            nn.Linear(stats_in_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.substitute_proj = nn.Linear(embedding_dim, embedding_dim)

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

    def _lightgcn_propagate(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.user_embedding.weight.device
        if adj.device != device:
            adj = adj.to(device)
        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego]
        x = ego
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            all_embeddings.append(x)
        final = torch.stack(all_embeddings, dim=0).mean(dim=0)
        user_emb, item_emb = torch.split(final, [self.num_users, self.num_items], dim=0)
        return user_emb, item_emb

    def _gather_behavior_embeddings(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        behavior_embeddings: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in self.behavior_names:
            behavior_embeddings[name] = self._lightgcn_propagate(self._get_behavior_adj(name))
        return behavior_embeddings

    def _stats_score(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Apply statistics network on stacked inputs."""
        joint = torch.cat([z, e, z * e], dim=-1)
        return self.statistics_net(joint).squeeze(-1)

    def _aggregate_intent(
        self,
        target_item: torch.Tensor,
        target_user: torch.Tensor,
        aux_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if aux_embeddings is None:
            return self.intent_proj(target_user)
        query = self.attn_query(target_item).unsqueeze(1)
        key = self.attn_key(aux_embeddings)
        value = self.attn_value(aux_embeddings)
        attn_logits = torch.sum(query * key, dim=-1) / math.sqrt(self.embedding_dim)
        attn_weights = torch.softmax(attn_logits, dim=1)
        aggregated = torch.sum(attn_weights.unsqueeze(-1) * value, dim=1)
        return self.intent_proj(aggregated + target_user)

    def _score_items(self, intent: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return torch.sum(intent.unsqueeze(1) * items, dim=-1)

    def forward(
        self,
        user_indices: torch.Tensor,
        pos_item_indices: torch.Tensor,
        neg_item_indices: torch.Tensor,
        hesitation_item_indices: Optional[torch.Tensor] = None,
        purchased_item_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        user_idx = self._ensure_long_tensor(user_indices)
        pos_idx = self._ensure_long_tensor(pos_item_indices)
        neg_idx = self._ensure_long_tensor(neg_item_indices)

        behavior_embeddings = self._gather_behavior_embeddings()
        target_user_all, target_item_all = behavior_embeddings[self.target_behavior]
        user_target = target_user_all[user_idx]
        pos_items = target_item_all[pos_idx]
        neg_items = target_item_all[neg_idx]

        if self.aux_behaviors:
            aux_stack = []
            for behavior in self.aux_behaviors:
                aux_user_all, _ = behavior_embeddings[behavior]
                aux_stack.append(aux_user_all[user_idx])
            aux_embeddings = torch.stack(aux_stack, dim=1)
        else:
            aux_embeddings = None

        intent = self._aggregate_intent(pos_items, user_target, aux_embeddings)
        target_interaction = user_target * pos_items

        bsz = user_target.size(0)
        mi_intents = intent.unsqueeze(1).expand(-1, bsz, -1)
        mi_items = target_interaction.unsqueeze(0).expand(bsz, -1, -1)
        mi_scores = self._stats_score(mi_intents, mi_items)

        pos_scores = torch.sum(intent * pos_items, dim=-1)
        neg_scores = torch.sum(intent * neg_items, dim=-1)

        hes_mask: Optional[torch.Tensor]
        if hesitation_item_indices is not None:
            hes_idx = self._ensure_long_tensor(hesitation_item_indices)
            hes_mask = hes_idx.ge(0)
            hes_idx_safe = hes_idx.clamp(min=0)
            hes_emb = target_item_all[hes_idx_safe].view(hes_idx.shape + (self.embedding_dim,))
            hes_emb = torch.where(hes_mask.unsqueeze(-1), hes_emb, torch.zeros_like(hes_emb))
            cf_items = hes_emb + self.injection_alpha * user_target.unsqueeze(1)
            cf_interactions = user_target.unsqueeze(1) * cf_items
            stats_input_intent = intent.unsqueeze(1).expand_as(cf_items)
            hesitation_stats = self._stats_score(stats_input_intent, cf_interactions)
            hesitation_confidence = torch.sigmoid(hesitation_stats / self.hesitation_temperature)
            hesitation_confidence = torch.where(hes_mask, hesitation_confidence, torch.zeros_like(hesitation_confidence))
            hesitation_scores = torch.sum(intent.unsqueeze(1) * cf_items, dim=-1)
        else:
            hes_mask = None
            hesitation_scores = None
            hesitation_confidence = None
            hesitation_stats = None
            cf_items = None

        purchased_mask: Optional[torch.Tensor]
        if purchased_item_indices is not None:
            purchased_idx = self._ensure_long_tensor(purchased_item_indices)
            purchased_mask = purchased_idx.ge(0)
            purchased_idx_safe = purchased_idx.clamp(min=0)
            purchased_emb = target_item_all[purchased_idx_safe].view(purchased_idx.shape + (self.embedding_dim,))
            purchased_emb = torch.where(
                purchased_mask.unsqueeze(-1), purchased_emb, torch.zeros_like(purchased_emb)
            )
            purchased_scores = torch.sum(intent.unsqueeze(1) * purchased_emb, dim=-1)
        else:
            purchased_mask = None
            purchased_emb = None
            purchased_scores = None

        substitution_similarity = None
        if (
            hesitation_scores is not None
            and cf_items is not None
            and cf_items.size(1) > 0
            and purchased_item_indices is not None
            and purchased_emb is not None
            and purchased_emb.size(1) > 0
        ):
            cand_proj = F.normalize(self.substitute_proj(cf_items), dim=-1, eps=1e-8)
            purchased_proj = F.normalize(self.substitute_proj(purchased_emb), dim=-1, eps=1e-8)
            sim = torch.einsum("bkd,bpd->bkp", cand_proj, purchased_proj)
            valid_pairs = hes_mask.unsqueeze(-1) & purchased_mask.unsqueeze(-2)
            sim = torch.where(valid_pairs, sim, torch.zeros_like(sim))
            if sim.size(-1) > 0:
                max_sim, _ = sim.max(dim=-1)
            else:
                max_sim = torch.zeros_like(hesitation_scores)
            substitution_similarity = torch.where(
                hes_mask,
                torch.relu(max_sim - self.substitution_threshold),
                torch.zeros_like(hesitation_scores),
            )
            decay = torch.exp(-self.substitution_penalty * substitution_similarity)
            hesitation_confidence = hesitation_confidence * decay

        return {
            "intent_embeddings": intent,
            "target_interactions": target_interaction,
            "mi_scores": mi_scores,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "hesitation_scores": hesitation_scores,
            "hesitation_confidence": hesitation_confidence,
            "hesitation_stats": hesitation_stats,
            "hesitation_mask": hes_mask,
            "counterfactual_items": cf_items,
            "purchased_scores": purchased_scores,
            "purchased_mask": purchased_mask,
            "substitution_similarity": substitution_similarity,
        }

    def score_pairs(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_idx = self._ensure_long_tensor(user_indices)
        item_idx = self._ensure_long_tensor(item_indices)
        behavior_embeddings = self._gather_behavior_embeddings()
        target_user_all, target_item_all = behavior_embeddings[self.target_behavior]
        user_target = target_user_all[user_idx]
        pos_items = target_item_all[item_idx]
        if self.aux_behaviors:
            aux_stack = []
            for behavior in self.aux_behaviors:
                aux_user_all, _ = behavior_embeddings[behavior]
                aux_stack.append(aux_user_all[user_idx])
            aux_embeddings: Optional[torch.Tensor] = torch.stack(aux_stack, dim=1)
        else:
            aux_embeddings = None
        intent = self._aggregate_intent(pos_items, user_target, aux_embeddings)
        scores = torch.sum(intent * pos_items, dim=-1)
        return scores

    @torch.no_grad()
    def predict(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(user_indices, item_indices)


class CHIMELoss(nn.Module):
    """Implements Eq. (14) & (15): MI + substitute penalty + tri-level BPR."""

    def __init__(
        self,
        info_nce_temperature: float = 0.2,
        general_bpr_weight: float = 1.0,
        hesitation_bpr_weight: float = 0.6,
        hesitation_weight: float = 1.2,
        substitution_margin: float = 0.05,
        substitution_weight: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.info_nce_temperature = info_nce_temperature
        self.general_bpr_weight = general_bpr_weight
        self.hesitation_bpr_weight = hesitation_bpr_weight
        self.hesitation_weight = hesitation_weight
        self.substitution_margin = substitution_margin
        self.substitution_weight = substitution_weight
        self.eps = eps

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        info_loss = self._info_nce(outputs["mi_scores"])
        rec_loss = self._tri_bpr(outputs)
        substitution_loss = self._substitution_penalty(outputs)
        total = (
            self.general_bpr_weight * rec_loss["general"]
            + self.hesitation_bpr_weight * rec_loss["hesitation"]
            + self.substitution_weight * substitution_loss
            + info_loss
        )
        return {
            "total": total,
            "info": info_loss,
            "general_bpr": rec_loss["general"],
            "hesitation_bpr": rec_loss["hesitation"],
            "substitution": substitution_loss,
        }

    def _info_nce(self, mi_scores: torch.Tensor) -> torch.Tensor:
        logits = mi_scores / self.info_nce_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_i + loss_j)

    def _tri_bpr(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_scores = outputs["pos_scores"]
        neg_scores = outputs["neg_scores"]
        general_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        hesitation_scores = outputs.get("hesitation_scores")
        hesitation_confidence = outputs.get("hesitation_confidence")
        hesitation_mask = outputs.get("hesitation_mask")

        if hesitation_scores is None or hesitation_confidence is None:
            return {"general": general_loss, "hesitation": pos_scores.new_tensor(0.0)}

        mask = hesitation_mask if hesitation_mask is not None else torch.ones_like(hesitation_scores, dtype=torch.bool)
        pos_diff = pos_scores.unsqueeze(1) - hesitation_scores
        hes_diff = hesitation_scores - neg_scores.unsqueeze(1)
        loss_pos_hes = -F.logsigmoid(pos_diff)
        loss_hes_neg = -F.logsigmoid(hes_diff)
        weighted_hes_neg = loss_hes_neg * hesitation_confidence

        masked_pos_hes = loss_pos_hes.masked_select(mask)
        masked_hes_neg = weighted_hes_neg.masked_select(mask)

        hes_loss_pos = masked_pos_hes.mean() if masked_pos_hes.numel() > 0 else pos_scores.new_tensor(0.0)
        hes_loss_neg = masked_hes_neg.mean() if masked_hes_neg.numel() > 0 else pos_scores.new_tensor(0.0)
        hesitation_loss = hes_loss_pos + self.hesitation_weight * hes_loss_neg
        return {"general": general_loss, "hesitation": hesitation_loss}

    def _substitution_penalty(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hesitation_scores = outputs.get("hesitation_scores")
        substitution_similarity = outputs.get("substitution_similarity")
        purchased_scores = outputs.get("purchased_scores")
        purchased_mask = outputs.get("purchased_mask")

        if (
            hesitation_scores is None
            or substitution_similarity is None
            or purchased_scores is None
            or purchased_mask is None
        ):
            device = None
            if hesitation_scores is not None:
                device = hesitation_scores.device
            elif purchased_scores is not None:
                device = purchased_scores.device
            else:
                device = torch.device("cpu")
            return torch.zeros((), device=device)

        masked_scores = torch.where(
            purchased_mask, purchased_scores, torch.full_like(purchased_scores, float("-inf"))
        )
        max_purchased = masked_scores.max(dim=1).values
        has_valid = purchased_mask.any(dim=1)
        max_purchased = torch.where(has_valid, max_purchased, torch.zeros_like(max_purchased))

        margin_term = self.substitution_margin + hesitation_scores - max_purchased.unsqueeze(1)
        hinge = torch.relu(margin_term)
        weighted = hinge * substitution_similarity
        denom = substitution_similarity.sum() + self.eps
        return weighted.sum() / denom
