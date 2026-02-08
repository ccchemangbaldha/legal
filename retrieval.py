import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from embedder import embed_text

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def extract_terms(q):
    return re.findall(r"article\s+\d+", q.lower())


def retrieve(query):

    vec = embed_text(query)

    res = index.query(
        vector=vec,
        top_k=12,
        include_metadata=True
    )

    terms = extract_terms(query)

    scored = []

    for m in res["matches"]:
        text = m["metadata"].get("text", "")
        score = m["score"]

        for t in terms:
            if t in text:
                score += 0.25

        scored.append((score, m))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scored[:5]]
