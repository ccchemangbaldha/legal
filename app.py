import streamlit as st
import tempfile

from chunker import load_and_chunk
from embedder import embed_text

st.set_page_config(layout="wide")
st.title("📄 Legal PDF Chunk + Embedding Viewer")

uploaded = st.file_uploader("Upload PDF")

if uploaded:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded.read())
    tmp.close()

    with st.spinner("Chunking document..."):
        chunks = load_and_chunk(tmp.name)

    st.success(f"Total chunks: {len(chunks)}")

    st.divider()

    for idx, ch in enumerate(chunks):

        with st.expander(f"Chunk {idx+1} — Page {ch['page']} — Part {ch['part']}"):

            st.write("**Token count:**", ch["tokens"])

            if st.button(f"Generate embedding #{idx}", key=idx):
                vec = embed_text(ch["text"])
                st.write("Embedding dimension:", len(vec))
                st.write(vec[:12], "...")

            st.text_area(
                "Chunk Text",
                ch["text"],
                height=220
            )
