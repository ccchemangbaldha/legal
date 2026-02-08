import streamlit as st
import tempfile
import time

from chunker import load_and_chunk
from embedder import embed_text
from pinecone_store import ensure_index, upsert_chunks
from retrieval import retrieve
from llm_answer import answer

# 1. Page Config
st.set_page_config(layout="wide", page_title="Legal RAG Chat")
st.title("⚖️ Legal RAG Assistant")

# 2. Sidebar: Document Ingestion
with st.sidebar:
    st.header("📂 Data Ingestion")
    st.markdown("Upload a legal PDF to knowledge base.")
    
    pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name

        # Process chunks
        with st.spinner("Chunking PDF..."):
            chunks = load_and_chunk(tmp_path)
        
        st.success(f"Ready: {len(chunks)} chunks generated.")

        # Store button
        if st.button("Store in Pinecone", type="primary"):
            with st.spinner("Indexing data..."):
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
                    st.toast(f"✅ Successfully stored {len(vecs)} vectors!", icon="🎉")
                    time.sleep(1)
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown("### ℹ️ How to use")
    st.caption("1. Upload a PDF.\n2. Click 'Store in Pinecone'.\n3. Chat with your document.")

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display Evidence if available
        if "evidence" in message:
            with st.expander("🔍 View Retrieved Evidence"):
                for item in message["evidence"]:
                    st.markdown(f"**Page {item['page']}**")
                    st.caption(item["text"])
                    st.divider()
        
        # Display Token Usage if available
        if "usage" in message:
            st.caption(f"📊 **Usage:** {message['usage']}")

# 5. Chat Input & Response Logic
if query := st.chat_input("Ask a question about your legal documents..."):
    
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal texts..."):
            
            # Retrieve
            hits = retrieve(query)

            if not hits:
                response = "I couldn't find any relevant information. Please ensure you have uploaded and stored a document in the sidebar."
                st.warning(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            else:
                # LLM Answer
                ans_text, usage_obj = answer(query, hits)
                
                # Display Answer
                st.markdown(ans_text)

                # Prepare Evidence Data for History
                evidence_data = []
                with st.expander("🔍 View Retrieved Evidence"):
                    for h in hits:
                        md = h["metadata"]
                        evidence_data.append({"page": md["page"], "text": md["text"]})
                        st.markdown(f"**Page {md['page']}**")
                        st.caption(f"{md['text'][:300]}...") # Show snippet
                        st.divider()

                # Prepare Usage String
                usage_str = f"Input: {usage_obj.prompt_tokens} | Output: {usage_obj.completion_tokens} | Total: {usage_obj.total_tokens}"
                st.caption(f"📊 **Usage:** {usage_str}")

                # Save to History
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans_text,
                    "evidence": evidence_data,
                    "usage": usage_str
                })