from elasticsearch.helpers import bulk
from elastic_index import get_client, ensure_index, ES_INDEX
import re


def extract_article(text: str):
    m = re.search(r"article\s+(\d+)", text.lower())
    if m:
        return f"article_{m.group(1)}"
    return None


def build_actions(chunks, source_name):
    for ch in chunks:
        text = ch["text"]
        yield {
            "_index": ES_INDEX,
            "_id": f"{source_name}_p{ch['page']}_{ch['part']}",
            "_source": {
                "text": text,
                "source": source_name,
                "page": ch["page"],
                "part": ch["part"],
                "article": extract_article(text),
                "title": text[:120]
            }
        }


def bulk_upsert(chunks, source_name):
    ensure_index()
    es = get_client()
    actions = list(build_actions(chunks, source_name))
    bulk(es, actions)
    print(f"Indexed {len(actions)} docs into Elasticsearch")

