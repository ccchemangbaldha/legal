import streamlit as st
import tempfile

from chunker import load_and_chunk
from embedder import embed_text
from pinecone_store import ensure_index, upsert_chunks
from retrieval import retrieve
from llm_answer import answer

st.set_page_config(layout="wide")
st.title("⚖️ Legal RAG System")

# -------- INGEST --------

pdf = st.file_uploader("Upload PDF")

if pdf:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.read())
    tmp.close()

    chunks = load_and_chunk(tmp.name)
    st.success(f"Chunks: {len(chunks)}")

    if st.button("Store in Pinecone"):

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

        upsert_chunks(vecs, 10)
        st.success("Stored ✅")

# -------- ASK --------

st.divider()
q = st.text_input("Ask question")

if q:

    hits = retrieve(q)

    st.subheader("Evidence")

    for h in hits:
        md = h["metadata"]
        st.markdown(f"**Page {md['page']} Part {md['part']}**")
        st.text_area("", md["text"], height=160)

    ans, usage = answer(q, hits)

    st.subheader("Answer")
    st.write(ans)

    st.write("Tokens:", {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "total": usage.total_tokens
    })
