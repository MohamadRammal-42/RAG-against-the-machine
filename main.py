import bm25s
from splitter import get_files, extract_chunks
from sentence_transformers import CrossEncoder, SentenceTransformer
from functools import lru_cache
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    k = 5
    Data, List_texts, units = get_files("./vllm-0.10.1")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        units, batch_size=64, normalize_embeddings=True,
        show_progress_bar=True
    )
    np.save("data/processed/embeddings.npy", embeddings)

    tokens = bm25s.tokenize(List_texts)
    retriever = bm25s.BM25()
    retriever.index(tokens)

    retriever.save("data/processed/bm25")

    @lru_cache(None)
    def _retreiving(question):
        ambedding_query = embedder.encode(
            question, normalize_embeddings=True
        )
        retriever = bm25s.BM25.load("data/processed/bm25")
        question_tokens = bm25s.tokenize(question)
        results, scores = retriever.retrieve(question_tokens, k=(min(max(20, k * 5), len(List_texts))))
        candidates = [List_texts[idx] for idx in results[0]]
        # rerank = CrossEncoder("BAAI/bge-reranker-base")
        # scores = rerank.predict([(question, chunk) for chunk in candidates])
        candidates_chunk = extract_chunks(Data, candidates)

        ranked = list(zip(candidates, scores))
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)


    questions = ["What activation formats does the fused batched MoE layer return in vLLM?"]

    for question in questions:
        _retreiving(question)
