# ⚖️ Legal RAG Assistant - Project Documentation

## 📖 Overview
The **Legal RAG Assistant** is a Retrieval-Augmented Generation (RAG) application designed specifically for legal documents. It allows users to upload legal texts (PDF, TXT), index them using a hybrid approach (Vector + Keyword), and ask precise legal questions. The system leverages **Pinecone** for semantic search, **Elasticsearch** for keyword matching, and **OpenAI's GPT models** to generate legally accurate answers with citations.

---

## 🏗️ System Architecture

The application follows a standard RAG pipeline with a Hybrid Retrieval twist:

1.  **Ingestion Layer**:
    * **Chunking**: Splits documents into manageable segments using a sliding window approach to preserve context.
    * **Embedding**: Converts text chunks into vector representations using the `law-ai/InLegalBERT` model.
    * **Storage**:
        * **Vectors** -> Stored in **Pinecone**.
        * **Keywords/Metadata** -> Stored in **Elasticsearch**.

2.  **Retrieval Layer (Hybrid)**:
    * Incoming queries are processed by both search engines.
    * **Vector Search**: Finds semantically similar chunks.
    * **Keyword Search**: Finds exact matches (boosting specific legal terms like "Article" or "Schedule").
    * **Merging**: Results are normalized and combined using a weighted score (Alpha).

3.  **Generation Layer**:
    * The top retrieved chunks are passed as context to the **OpenAI Assistant**.
    * The model generates an answer strictly based on the provided context.

4.  **Frontend**:
    * Built with **Streamlit** for an interactive chat and file upload interface.

---

## 📂 Codebase Structure

| File | Description |
| :--- | :--- |
| `app.py` | Main entry point. Streamlit UI for chat and data ingestion. |
| `chunker.py` | Logic for loading files (PDF/TXT) and splitting them into chunks. |
| `embedder.py` | Handles text embedding using the `InLegalBERT` HuggingFace model. |
| `pinecone_store.py` | Manages index creation and upserting data to Pinecone. |
| `elastic_store.py` | Manages index creation and bulk upserting to Elasticsearch. |
| `retrieval.py` | Performs vector search with specific keyword boosting logic. |
| `elastic_search.py` | Performs standard keyword/text search in Elasticsearch. |
| `retrieval_hybrid.py` | Orchestrates the combination of Vector and Keyword search results. |
| `llm_answer.py` | Interfaces with the OpenAI Assistants API to generate responses. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* Accounts & API Keys for:
    * **OpenAI**
    * **Pinecone** (Serverless)
    * **Elasticsearch** (Cloud or Local)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**:
    Create a `.env` file in the root directory. **Crucial:** Ensure `PINECONE_INDEX` is set to the same value to avoid index mismatch errors.

    ```ini
    # OpenAI
    OPENAI_API_KEY=sk-...
    OPENAI_ASSISTANT_ID=... (Optional: system creates one if missing)

    # Pinecone
    PINECONE_API_KEY=...
    PINECONE_REGION=us-east-1
    PINECONE_INDEX=legal-knowledge-base  # <--- MUST BE CONSISTENT

    # Elasticsearch
    ES_URL=...
    ES_API_KEY=...
    ES_INDEX=legal_chunks
    ```

### Running the Application
```bash
streamlit run app.py