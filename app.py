import streamlit as st
import tempfile
import time

from chunker import load_and_chunk
from embedder import embed_text
from pinecone_store import ensure_index, upsert_chunks
from retrieval import retrieve
from llm_answer import answer

st.set_page_config(layout="wide", page_title="Legal RAG")
st.title("⚖️ Legal RAG System")

# -------- INGEST --------

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF..."):
        chunks = load_and_chunk(tmp_path)
    
    st.info(f"Generated {len(chunks)} chunks.")

    if st.button("Store in Pinecone"):
        with st.spinner("Creating index and storing data..."):
            try:
                ensure_index()
                
                vecs = []
                for ch in chunks:
                    vid = f"{pdf.name}_p{ch['page']}_{ch['part']}"
                    vec = embed_text(ch["text"])
                    meta = {
                        "page": ch["page"],
                        "part": ch["part"],
                        "tokens": ch["tokens"],
                        "source": pdf.name,
                        "text": ch["text"]
                    }
                    vecs.append((vid, vec, meta))

                upsert_chunks(vecs, batch_size=20)
                st.success(f"Stored {len(vecs)} vectors! ✅")
                time.sleep(2) 
            except Exception as e:
                st.error(f"Error: {e}")

# -------- ASK --------

st.divider()
q = st.text_input("Ask a legal question")

if q:
    with st.spinner("Retrieving..."):
        hits = retrieve(q)

    if not hits:
        st.warning("No results found. Did you click 'Store in Pinecone'?")
    else:
        st.subheader("Evidence")
        cols = st.columns(6)
        for i, h in enumerate(hits[:6]):
            md = h["metadata"]
            with cols[i]:
                st.caption(f"Page {md['page']}")
                st.markdown(f"_{md['text'][:150]}..._")

        ans, usage = answer(q, hits)

        st.subheader("Answer")
        st.write(ans)