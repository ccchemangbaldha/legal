import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "legal-chunks")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)


def ensure_index(dimension=768):

    existing = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )


def upsert_chunks(vectors, batch_size=40):
    """
    vectors = [(id, vector, metadata)]
    """

    index = pc.Index(PINECONE_INDEX)

    payload = [
        {
            "id": vid,
            "values": vec,
            "metadata": meta
        }
        for vid, vec, meta in vectors
    ]

    for i in range(0, len(payload), batch_size):
        batch = payload[i:i + batch_size]
        index.upsert(batch)
        print(f"Upserted batch {i//batch_size + 1}")
