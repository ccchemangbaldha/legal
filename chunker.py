import re
import os
from pypdf import PdfReader
from embedder import token_len

# KEEPING SKIP_PAGES LOW to ensure we get Article 13 etc.
SKIP_PAGES = 0
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

def load_and_chunk(file_path: str):
    """
    Detects file type based on extension and chunks accordingly.
    Supports .pdf, .txt, .log
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    chunks = []

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            if i < SKIP_PAGES:
                continue

            raw = page.extract_text() or ""
            text = clean_text(raw)

            if not text:
                continue

            batch_chunks = split_text_sliding_window(text, chunk_size_words=300, overlap_words=50)

            for idx, chunk_text in enumerate(batch_chunks):
                chunks.append({
                    "page": i + 1,
                    "part": f"batch_{idx}",
                    "text": chunk_text,
                    "tokens": token_len(chunk_text)
                })

    elif ext in [".txt", ".log"]:
        # Text based files
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return []

        text = clean_text(raw)
        if text:
            batch_chunks = split_text_sliding_window(text, chunk_size_words=300, overlap_words=50)
            
            for idx, chunk_text in enumerate(batch_chunks):
                chunks.append({
                    "page": 1, # Treat entire text file as page 1
                    "part": f"batch_{idx}",
                    "text": chunk_text,
                    "tokens": token_len(chunk_text)
                })

    return chunks