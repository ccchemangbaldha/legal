from retrieval import retrieve as pinecone_retrieve
from elastic_search import search_es


def normalize(scores):
    if not scores:
        return {}
    max_s = max(scores.values())
    if max_s == 0:
        return scores
    return {k: v / max_s for k, v in scores.items()}


def hybrid_retrieve(query, k=5, alpha=0.6):
    """
    alpha = semantic weight
    (1-alpha) = keyword weight
    """

    pine_hits = pinecone_retrieve(query)
    es_hits = search_es(query, top_k=10)

    pine_scores = {h["id"]: h["score"] for h in pine_hits}
    es_scores = {h["id"]: h["score"] for h in es_hits}

    pine_scores = normalize(pine_scores)
    es_scores = normalize(es_scores)

    all_ids = set(pine_scores) | set(es_scores)

    merged = []

    pine_map = {h["id"]: h for h in pine_hits}
    es_map = {h["id"]: h for h in es_hits}

    for vid in all_ids:
        ps = pine_scores.get(vid, 0)
        ks = es_scores.get(vid, 0)
        score = alpha * ps + (1 - alpha) * ks

        item = pine_map.get(vid) or es_map.get(vid)
        merged.append((score, item))

    merged.sort(reverse=True, key=lambda x: x[0])

    return [m for _, m in merged[:k]]
