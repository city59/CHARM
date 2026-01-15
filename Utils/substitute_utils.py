from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np
import torch as t


def _ensure_iterable(items: Optional[Iterable[int]]) -> Set[int]:
    if items is None:
        return set()
    if isinstance(items, set):
        return items
    if isinstance(items, list):
        return set(items)
    if isinstance(items, tuple):
        return set(items)
    return set(items)


def build_latent_substitute_map(
    user_history_dict: Dict[int, Dict[str, Sequence[int]]],
    interest_keys: Optional[Sequence[str]] = None,
    purchase_key: str = "buy",
    jaccard_threshold: float = 0.1,
    max_copurchase: int = 0,
) -> Dict[int, Set[int]]:
    interest_users: Dict[int, Set[int]] = defaultdict(set)
    purchase_users: Dict[int, Set[int]] = defaultdict(set)
    pair_intersections: Dict[tuple, int] = defaultdict(int)

    for user, behaviors in user_history_dict.items():
        if not isinstance(behaviors, dict):
            continue
        if interest_keys is None:
            candidate_keys = [k for k in behaviors.keys() if k != purchase_key]
        else:
            candidate_keys = interest_keys

        interest_items = set()
        for key in candidate_keys:
            items = behaviors.get(key)
            if not items:
                continue
            interest_items.update(_ensure_iterable(items))

        interest_list: List[int] = sorted(interest_items)
        for item in interest_list:
            interest_users[item].add(user)

        for item_i, item_j in combinations(interest_list, 2):
            pair_intersections[(item_i, item_j)] += 1

        if purchase_key is not None:
            raw_purchase = behaviors.get(purchase_key)
            if raw_purchase:
                for item in _ensure_iterable(raw_purchase):
                    purchase_users[item].add(user)

    substitute_map: Dict[int, Set[int]] = defaultdict(set)
    for (item_i, item_j), intersection in pair_intersections.items():
        users_i = interest_users.get(item_i)
        users_j = interest_users.get(item_j)
        if not users_i or not users_j:
            continue
        union = len(users_i) + len(users_j) - intersection
        if union <= 0:
            continue
        jaccard = intersection / union
        if jaccard < jaccard_threshold:
            continue

        copurchase_count = len(purchase_users.get(item_i, set()) & purchase_users.get(item_j, set()))
        if copurchase_count > max_copurchase:
            continue

        substitute_map[item_i].add(item_j)
        substitute_map[item_j].add(item_i)

    return {item: subs for item, subs in substitute_map.items() if subs}


def _flatten_id_array(
    values: Optional[Union[Sequence[int], np.ndarray, t.Tensor]],
    expected_len: int,
) -> List[int]:
    if values is None:
        flat: List[int] = []
    elif isinstance(values, t.Tensor):
        flat = t.as_tensor(values).detach().reshape(-1).cpu().tolist()
    elif isinstance(values, np.ndarray):
        arr = values.reshape(-1)
        flat = arr.astype(np.int64, copy=False).tolist()
    else:
        seq = list(values)
        if not seq:
            flat = []
        else:
            arr = np.asarray(seq).reshape(-1)
            flat = arr.astype(np.int64, copy=False).tolist()
    if len(flat) != expected_len:
        raise ValueError(f"Expected {expected_len} ids, but received {len(flat)}")
    return [int(v) for v in flat]


def apply_fulfillment_correction(
    scores: t.Tensor,
    user_ids: Union[Sequence[int], np.ndarray, t.Tensor],
    item_ids: Union[Sequence[int], np.ndarray, t.Tensor],
    user_history: Dict[int, Set[int]],
    substitute_map: Dict[int, Set[int]],
    penalty: float = 0.2,
) -> t.Tensor:
    """
    Penalize scores for user-item pairs where the user already bought a substitute item.
    """
    if not isinstance(scores, t.Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if penalty <= 0 or not user_history or not substitute_map:
        return scores

    scores_flat = scores.reshape(-1)
    num_pairs = scores_flat.numel()
    user_list = _flatten_id_array(user_ids, num_pairs)
    item_list = _flatten_id_array(item_ids, num_pairs)

    penalize_flags = []
    for uid, iid in zip(user_list, item_list):
        purchased = user_history.get(uid)
        substitutes = substitute_map.get(iid)
        penalize_flags.append(bool(purchased and substitutes and (purchased & substitutes)))

    if not any(penalize_flags):
        return scores

    mask = t.as_tensor(penalize_flags, dtype=scores_flat.dtype, device=scores.device)
    scaling = 1 - penalty * mask
    corrected = scores_flat * scaling
    return corrected.view_as(scores)
