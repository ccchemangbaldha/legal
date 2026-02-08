import re
from pypdf import PdfReader
from embedder import token_len

SKIP_PAGES = 30
TOKEN_LIMIT = 450


def clean_text(t: str) -> str:
    if not t:
        return ""

    # remove long dash / underscore / star separator lines
    t = re.sub(r"[-_*]{3,}", " ", t)

    # remove repeated dots
    t = re.sub(r"[.]{3,}", " ", t)

    # remove weird box characters
    t = re.sub(r"[│┤┘┐┌└├┬┴┼]", " ", t)

    # normalize newlines → space
    t = t.replace("\n", " ")

    # collapse whitespace
    t = re.sub(r"\s+", " ", t)

    # strip edges
    t = t.strip()

    # lowercase for keyword match friendliness
    t = t.lower()

    return t


def split_if_needed(text: str):

    if token_len(text) <= TOKEN_LIMIT:
        return [("full", text)]

    words = text.split()
    half = len(words) // 2

    return [
        ("a", " ".join(words[:half])),
        ("b", " ".join(words[half:]))
    ]


def load_and_chunk(pdf_path: str):

    reader = PdfReader(pdf_path)
    chunks = []

    for i, page in enumerate(reader.pages):

        if i < SKIP_PAGES:
            continue

        raw = page.extract_text() or ""
        text = clean_text(raw)

        if not text:
            continue

        parts = split_if_needed(text)

        for part, chunk_text in parts:

            chunks.append({
                "page": i + 1,
                "part": part,
                "text": chunk_text,
                "tokens": token_len(chunk_text)
            })

    return chunks
