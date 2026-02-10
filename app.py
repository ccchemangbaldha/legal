import streamlit as st
import tempfile
import time
import os

from chunker import load_and_chunk
from embedder import embed_text
from pinecone_store import ensure_index as ensure_pinecone_index, upsert_chunks
from elastic_store import bulk_upsert as es_bulk_upsert, ensure_index as ensure_es_index
from retrieval_hybrid import hybrid_retrieve as retrieve
from llm_answer import answer

st.set_page_config(layout="wide", page_title="Legal RAG Chat")
st.title("⚖️ Legal RAG Assistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

with st.sidebar:
    st.header("📂 Data Ingestion")
    st.markdown("Upload a document (PDF, TXT, LOG) to knowledge base.")

    # Changed type to include txt and log
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "log"])

    if uploaded_file:
        # Determine the file extension to save correctly
        file_ext = os.path.splitext(uploaded_file.name)[1]
        
        # Save with the correct extension so chunker knows how to handle it
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(f"Chunking {file_ext} file..."):
            chunks = load_and_chunk(tmp_path)

        st.success(f"Ready: {len(chunks)} chunks generated.")

        if st.button("Store in Vector + Keyword Index", type="primary"):
            with st.spinner("Indexing into Pinecone + Elasticsearch..."):
                try:
                    ensure_pinecone_index()
                    ensure_es_index()

                    vecs = []
                    for ch in chunks:
                        # Updated ID generation to use uploaded_file.name
                        vid = f"{uploaded_file.name}_p{ch['page']}_{ch['part']}"
                        vec = embed_text(ch["text"])
                        meta = {
                            "page": ch["page"],
                            "part": ch["part"],
                            "tokens": ch["tokens"],
                            "source": uploaded_file.name,
                            "text": ch["text"]
                        }
                        vecs.append((vid, vec, meta))

                    upsert_chunks(vecs, batch_size=20)

                    es_bulk_upsert(chunks, uploaded_file.name)

                    st.toast(
                        f"✅ Stored {len(vecs)} chunks in Pinecone + Elasticsearch!",
                        icon="🎉"
                    )
                    time.sleep(1)

                except Exception as e:
                    st.error(f"Indexing Error: {e}")

    st.divider()
    st.markdown("### ℹ️ How to use")
    st.caption(
        "1. Upload a Document.\n"
        "2. Click 'Store in Vector + Keyword Index'.\n"
        "3. Ask legal questions (Articles, Rules, Roles, etc.)."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "evidence" in message:
            with st.expander("🔍 View Retrieved Evidence"):
                for item in message["evidence"]:
                    st.markdown(f"**Page {item['page']}**")
                    st.caption(item["text"])
                    st.divider()

        if "usage" in message:
            st.caption(f"📊 **Usage:** {message['usage']}")

if query := st.chat_input("Ask a question about your legal documents..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Running hybrid legal retrieval..."):

            hits = retrieve(query)

            if not hits:
                response = (
                    "No relevant chunks found. Make sure the document is indexed "
                    "and Elasticsearch + Pinecone are configured correctly."
                )
                st.warning(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            else:
                ans_text, usage_obj, new_thread_id = answer(
                    question=query,
                    matches=hits,
                    thread_id=st.session_state.thread_id
                )

                st.session_state.thread_id = new_thread_id

                st.markdown(ans_text)

                evidence_data = []
                with st.expander("🔍 View Retrieved Evidence"):
                    for h in hits:
                        md = h["metadata"]
                        evidence_data.append({
                            "page": md.get("page"),
                            "text": md.get("text")
                        })
                        st.markdown(f"**Page {md.get('page')}**")
                        st.caption(f"{md.get('text','')[:300]}...")
                        st.divider()

                if usage_obj:
                    usage_str = (
                        f"Input: {usage_obj.prompt_tokens} | "
                        f"Output: {usage_obj.completion_tokens} | "
                        f"Total: {usage_obj.total_tokens}"
                    )
                else:
                    usage_str = "Usage data unavailable"

                st.caption(f"📊 **Usage:** {usage_str}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans_text,
                    "evidence": evidence_data,
                    "usage": usage_str
                })