from typing import List, Tuple

def reciprocal_rank_fusion(*ranked_lists: List[str], k: int = 60) -> List[Tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, uid in enumerate(ranked_list):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)