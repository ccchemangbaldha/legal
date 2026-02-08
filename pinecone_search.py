import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "second-index")
REGION = os.getenv("PINECONE_REGION", "us-east-1")


def ensure_index():

    names = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=REGION)
        )


def upsert_batch(vectors, batch_size=40):

    index = pc.Index(INDEX_NAME)

    payload = []
    for vid, vec, meta in vectors:

        # 🔒 double safety conversion
        vec = [float(x) for x in vec]

        payload.append({
            "id": vid,
            "values": vec,
            "metadata": meta
        })

    for i in range(0, len(payload), batch_size):
        index.upsert(payload[i:i+batch_size])
