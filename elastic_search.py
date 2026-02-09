from elastic_index import get_client, ES_INDEX


def search_es(query, top_k=10):
    es = get_client()

    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^3",
                    "article^4",
                    "text"
                ],
                "type": "best_fields"
            }
        }
    }

    res = es.search(index=ES_INDEX, body=body)

    hits = []
    for h in res["hits"]["hits"]:
        hits.append({
            "id": h["_id"],
            "score": h["_score"],
            "metadata": h["_source"]
        })

    return hits
