import re
from pypdf import PdfReader
from embedder import token_len

# KEEPING SKIP_PAGES LOW to ensure we get Article 13 etc.
SKIP_PAGES = 30
TOKEN_LIMIT = 450

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"[-_*]{3,}", " ", t)
    t = re.sub(r"[.]{3,}", " ", t)
    t = re.sub(r"[│┤┘┐┌└├┬┴┼]", " ", t)
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = t.strip().lower()
    return t

def split_text_sliding_window(text: str, chunk_size_words=300, overlap_words=50):
    """
    Splits text into chunks using a sliding window of words.
    Iterative approach (no recursion) to prevent crashes.
    """
    words = text.split()
    if not words:
        return []
        
    chunks = []
    i = 0
    
    # Loop until we reach the end of the text
    while i < len(words):
        # Define the window
        end_idx = i + chunk_size_words
        
        # Create the chunk
        chunk_words = words[i:end_idx]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Stop if we've reached the end
        if end_idx >= len(words):
            break
            
        # Move the window forward, keeping the overlap
        i += (chunk_size_words - overlap_words)
        
    return chunks

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

        # Use the new iterative batch splitter
        # 300 words is usually safe for a 450 token limit
        batch_chunks = split_text_sliding_window(text, chunk_size_words=300, overlap_words=50)

        for idx, chunk_text in enumerate(batch_chunks):
            chunks.append({
                "page": i + 1,
                "part": f"batch_{idx}",
                "text": chunk_text,
                "tokens": token_len(chunk_text)
            })

    return chunks