"""
evaluator.py - Real RAGAS evaluation for PhishingGuard-RAG.

Computes three RAGAS metrics:
  - Faithfulness
  - Answer Relevance
  - Context Relevance
"""
import os
from typing import Any, List


RAGAS_METRIC_KEYS = {
    "faithfulness": "faithfulness",
    "answer_relevance": "answer_relevancy",
    "context_relevance": "context_relevancy",
}


def _load_ragas():
    """Import RAGAS lazily so the app can start before deps are installed."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_relevancy,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Real RAGAS evaluation requires the project dependencies to be "
            "installed. Run `pip install -r requirements.txt` in this "
            "environment before evaluating."
        ) from exc

    return Dataset, evaluate, faithfulness, answer_relevancy, context_relevancy


def _ragas_runtime_kwargs() -> dict:
    """
    Optional local-Ollama judge configuration.

    By default RAGAS uses its configured default LLM/embeddings, normally
    OpenAI via OPENAI_API_KEY. Set RAGAS_PROVIDER=ollama to run the evaluator
    through the local Ollama endpoint with local HuggingFace embeddings.
    """
    provider = os.getenv("RAGAS_PROVIDER", "openai").strip().lower()
    if provider != "ollama":
        return {}

    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        from src.knowledge_base import EMBEDDING_MODEL
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS_PROVIDER=ollama requires langchain-community, "
            "sentence-transformers, and RAGAS LangChain wrappers installed."
        ) from exc

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("RAGAS_LLM_MODEL", os.getenv("LLM_MODEL", "mistral"))
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", EMBEDDING_MODEL)
    if embedding_model == "nomic-embed-text":
        # Backwards compatibility for older local-Ollama embedding settings.
        # This evaluator now uses HuggingFace embeddings, not Ollama embeddings.
        embedding_model = EMBEDDING_MODEL

    llm = ChatOllama(model=llm_model, base_url=base_url, temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return {
        "llm": LangchainLLMWrapper(llm),
        "embeddings": LangchainEmbeddingsWrapper(embeddings),
    }


def _extract_metric(result: Any, metric_key: str) -> float:
    """Read a single metric from RAGAS result objects across 0.1.x shapes."""
    value = None

    if isinstance(result, dict):
        value = result.get(metric_key)

    if value is None and hasattr(result, "__getitem__"):
        try:
            value = result[metric_key]
        except Exception:
            value = None

    if value is None and hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if metric_key in frame.columns and not frame.empty:
            value = frame[metric_key].iloc[0]

    if hasattr(value, "iloc"):
        value = value.iloc[0]
    elif isinstance(value, list):
        value = value[0] if value else 0.0

    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _format_ragas_error(exc: Exception) -> str:
    message = str(exc)
    provider = os.getenv("RAGAS_PROVIDER", "openai").strip().lower()
    llm_model = os.getenv("RAGAS_LLM_MODEL", os.getenv("LLM_MODEL", "mistral"))
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "")

    if "nomic-embed-text" in message:
        return (
            "RAGAS evaluation failed because the evaluator is still trying "
            "to use `nomic-embed-text` as a HuggingFace embedding model. "
            "Unset RAGAS_EMBEDDING_MODEL or restart Streamlit so the updated "
            "local HuggingFace embedding configuration is loaded."
        )

    if provider == "ollama" and "404" in message and llm_model in message:
        return (
            "RAGAS evaluation failed because Ollama could not load the "
            f"judge model `{llm_model}`. Run `ollama pull {llm_model}` "
            "or set RAGAS_LLM_MODEL to an installed Ollama chat model."
        )

    if provider == "ollama" and embedding_model and embedding_model in message:
        return (
            "RAGAS evaluation failed while loading the embedding model "
            f"`{embedding_model}`. Unset RAGAS_EMBEDDING_MODEL to use the "
            "project default `BAAI/bge-large-en-v1.5`, or set it to a valid "
            "HuggingFace/SentenceTransformer model."
        )

    if provider != "ollama" and "api_key" in message.lower():
        return (
            "RAGAS evaluation failed because no OpenAI API key is configured. "
            "Set OPENAI_API_KEY, or set RAGAS_PROVIDER=ollama to evaluate "
            "with the local Ollama judge."
        )

    return f"RAGAS evaluation failed: {message}"


def evaluate_response(
    question: str,
    answer: str,
    contexts: List[str],
) -> dict:
    """
    Compute real RAGAS metrics for a single RAG response.

    Returns scores on a 0-1 scale using the existing app-facing keys:
      faithfulness, answer_relevance, context_relevance
    """
    Dataset, evaluate, faithfulness, answer_relevancy, context_relevancy = (
        _load_ragas()
    )

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_relevancy],
            **_ragas_runtime_kwargs(),
        )
    except Exception as exc:
        raise RuntimeError(_format_ragas_error(exc)) from exc

    return {
        output_key: _extract_metric(result, ragas_key)
        for output_key, ragas_key in RAGAS_METRIC_KEYS.items()
    }


# ── Batch evaluation (for the Reports page) ──────────────────────────────────
TEST_QUERIES = [
    {
        "question": "How should we handle spear-phishing in an enterprise?",
    },
    {
        "question": "What are the indicators of a phishing email?",
    },
    {
        "question": "How does DMARC prevent email spoofing?",
    },
    {
        "question": "What steps should be taken after a phishing compromise?",
    },
    {
        "question": "What is Business Email Compromise?",
    },
]


def run_batch_evaluation(pipeline_fn) -> list:
    """Run the full pipeline on all test queries and collect RAGAS scores."""
    results = []
    for item in TEST_QUERIES:
        result = pipeline_fn(item["question"])
        scores = evaluate_response(
            item["question"],
            result["answer"],
            result["contexts"],
        )
        results.append({
            "question": item["question"],
            "answer": result["answer"],
            **scores,
        })
    return results
