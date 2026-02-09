import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_INDEX = os.getenv("ES_INDEX", "legal_chunks")


def get_client():
    return Elasticsearch(
        ES_URL,
        api_key=ES_API_KEY,
        request_timeout=60
    )


def ensure_index():
    es = get_client()

    if es.indices.exists(index=ES_INDEX):
        return

    mapping = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "legal_text_analyzer": {
                        "type": "standard"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "legal_text_analyzer"},
                "source": {"type": "keyword"},
                "page": {"type": "integer"},
                "part": {"type": "keyword"},
                "article": {"type": "keyword"},
                "title": {"type": "text"}
            }
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"Created ES index: {ES_INDEX}")

