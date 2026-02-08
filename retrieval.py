import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone, exceptions
from embedder import embed_text

load_dotenv()

# Initialize Client only (not Index yet)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "xxc")

def extract_terms(q):
    q = q.lower()
    specifics = re.findall(r"(article\s+\d+|schedule\s+\d+)", q)
    words = re.findall(r"\w{4,}", q)
    return list(set(specifics + words))

def retrieve(query):
    # 1. Connect to Index safely
    try:
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error connecting to index: {e}")
        return []

    # 2. Embed
    try:
        vec = embed_text(query)
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

    # 3. Query
    try:
        res = index.query(
            vector=vec,
            top_k=15,
            include_metadata=True
        )
    except exceptions.NotFoundException:
        print(f"Index '{INDEX_NAME}' not found.")
        return []
    except Exception as e:
        print(f"Query error: {e}")
        return []

    # 4. Keyword Boosting
    terms = extract_terms(query)
    scored = []

    for m in res.get("matches", []):
        text = m["metadata"].get("text", "").lower()
        score = m["score"]

        for t in terms:
            if t in text:
                if "article" in t or "schedule" in t:
                    score += 0.35 
                else:
                    score += 0.05

        scored.append((score, m))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scored[:5]]