import streamlit as st
import tempfile
import os

from chunker import load_and_chunk
from embedder import embed_text
from pinecone_store import ensure_index, upsert_chunks

st.set_page_config(layout="wide")
st.title("📄 Legal PDF Chunk + Embedding + Pinecone Store")

uploaded = st.file_uploader("Upload PDF")

if uploaded:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded.read())
    tmp.close()

    with st.spinner("Cleaning + chunking document..."):
        chunks = load_and_chunk(tmp.name)

    st.success(f"Total chunks: {len(chunks)}")

    store_vectors = st.button("Store All Chunks In Pinecone")

    vectors = []

    for idx, ch in enumerate(chunks):

        chunk_id = f"{uploaded.name}_p{ch['page']}_{ch['part']}"

        with st.expander(f"Chunk {idx+1} — Page {ch['page']} — Part {ch['part']}"):

            st.write("Token count:", ch["tokens"])

            if st.button(f"Embed #{idx}", key=f"emb{idx}"):
                vec = embed_text(ch["text"])
                st.write("Embedding dim:", len(vec))
                st.write(vec[:10])

            st.text_area("Chunk Text", ch["text"], height=200)

        if store_vectors:
            vec = embed_text(ch["text"])

            meta = {
                "page": ch["page"],
                "part": ch["part"],
                "tokens": ch["tokens"],
                "source": uploaded.name,
                "text": ch["text"]
            }

            vectors.append((chunk_id, vec.tolist(), meta))

    if store_vectors and vectors:
        with st.spinner("Creating index if needed..."):
            ensure_index()

        with st.spinner("Upserting to Pinecone..."):
            upsert_chunks(vectors, 10)

        st.success("Stored in Pinecone ✅")
