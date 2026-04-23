# PhishingGuard-RAG 🛡️

A domain-specific Retrieval-Augmented Generation system for phishing detection
guidance, built for COMP SCI / AI & ML Assignment 3.

## Architecture

```
User Query
    │
    ▼
Intent Interpreter (Mistral 7B)
    │  Reformulates ambiguous queries
    ▼
Hybrid Retriever (BM25 + FAISS)
    │  BAAI/bge-large-en-v1.5 embeddings
    │  Chunk size: 512 tokens, Overlap: 64 tokens
    ▼
LLM Generator (Mistral 7B)
    │  Context-grounded generation
    ▼
Ontology Verifier (Mistral 7B)
    │  Domain knowledge validation
    ▼
Validated Answer → Streamlit UI
    │
    ▼
RAGAS Evaluation (3 metrics) → SQLite
```

## Quick Start

### Prerequisites

1. **Install Ollama** — https://ollama.ai
2. **Pull Mistral 7B**:
   ```bash
   ollama pull mistral
   ```
3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
phishingguard_rag/
├── app.py                  # Streamlit main application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── src/
│   ├── knowledge_base.py   # Web crawler + FAISS indexer
│   ├── rag_pipeline.py     # Intent → Retrieval → Generation → Verification
│   ├── evaluator.py        # RAGAS metric computation
│   └── database.py         # SQLite persistence layer
├── knowledge_base/
│   ├── faiss_index/        # Auto-generated FAISS index
│   └── documents.pkl       # Auto-generated document store
└── data/
    └── phishingguard.db    # Auto-generated SQLite database
```

## Key Design Choices

| Component | Choice | Justification |
|-----------|--------|---------------|
| LLM | Mistral 7B (local) | Open-source, privacy-preserving, no API cost |
| Embedding | BAAI/bge-large-en-v1.5 | Top MTEB retrieval scores, local deployment |
| Vector DB | FAISS | Fast ANN search, fully local, no cloud dependency |
| Retrieval | Hybrid (BM25 + Dense) | Combines keyword precision + semantic recall |
| Evaluation | RAGAS (3 metrics) | Reference-free, component-level assessment |

## RAGAS Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Faithfulness | \|V\|/\|S\| | Fraction of answer claims supported by context |
| Answer Relevance | (1/n)Σsim(q,qᵢ) | How well the answer addresses the question |
| Context Relevance | extracted/total | Signal-to-noise ratio of retrieved context |

## Running Without Ollama (Demo Mode)

The system includes mock responses for testing without a running LLM.
Simply start the app — if Ollama is unavailable, mock responses are used
automatically.

## References

- Es et al. (2024). RAGAs: Automated Evaluation of RAG. EACL 2024.
- Lewis et al. (2020). RAG for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
- Jiang et al. (2023). Mistral 7B. arXiv:2310.06825.
- Verizon (2024). Data Breach Investigations Report.
- IBM Security (2024). Cost of a Data Breach Report.
