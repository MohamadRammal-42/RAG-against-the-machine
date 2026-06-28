import bm25s
from splitter import get_files, extract_chunks, get_chunks, extract_paragraphs
from sentence_transformers import CrossEncoder, SentenceTransformer
from functools import lru_cache
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

system_instruction = """
You are a technical documentation assistant.

Answer the user's question using only the provided context.

Rules:
- Return only the answer.
- Do not explain your reasoning.
- Do not restate the question.
- Do not include introductions or conversational phrases.
- If the answer is not contained in the context, reply exactly:
The answer is not available in the provided context.
"""

if __name__ == "__main__":
    k = 10
    Data, List_texts, units, chunk_id = get_files("./vllm-0.10.1")
    chunk_obj = get_chunks(Data)
    all_units = extract_paragraphs(chunk_obj)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = embedder.encode(
    #     units, batch_size=64, normalize_embeddings=True,
    #     show_progress_bar=True
    # )
    # np.save("data/processed/embeddings.npy", embeddings)
    embeddings = np.load("data/processed/embeddings.npy")

    tokens = bm25s.tokenize(List_texts)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save("data/processed/bm25")

    #Model installation
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    @lru_cache(None)
    def _bm25_retreiving(question):
        question_tokens = bm25s.tokenize(question)
        num = min(max(3 * k, 20), len(List_texts))
        results, scores = retriever.retrieve(question_tokens, k=num)
        candidates = [List_texts[idx] for idx in results[0]]
        candidates_chunk = extract_chunks(Data, candidates)

        ranked = list(zip(candidates_chunk, scores[0]))
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:num]]

    @lru_cache(None)
    def _embed_retreiver(question):
        ambedding_query = embedder.encode(
            question, normalize_embeddings=True
        )
        num = min(max(3 * k, 20), len(List_texts))
        scores = embeddings @ ambedding_query
        top_idx = np.argsort(scores)[::1][:num]
        return [chunk_id[i] for i in top_idx]

    def _rrf_calculation(question, k):
        bm_chunks = _bm25_retreiving(question)
        embedding_chunks = _embed_retreiver(question)

        rrf_ranked = {}

        for rank, chunk in enumerate(bm_chunks):
            rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k + rank)
        for rank, chunk in enumerate(embedding_chunks):
            rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k + rank)

        rrf_ranked = sorted(rrf_ranked.items(), key=lambda x: x[1], reverse=True)[:k]
        filterd_chunks = [0] * k
        rrf_ranked = [chunk for chunk, _ in rrf_ranked]
        for chunk in chunk_obj:
            if chunk["chunk_id"] in rrf_ranked:
                filterd_chunks[rrf_ranked.index(chunk["chunk_id"])] = chunk

        return filterd_chunks

    question = "What activation formats does the fused batched MoE layer return in vLLM?"
    import time
    start = time.time()
    print("start")

    filterd_chunks = _rrf_calculation(question, k)
    filtered_text = ""
    for chunk in filterd_chunks[:3]:
        filtered_text += chunk["text"]

    message = f"""
    Context:
    /no_think

    {filtered_text}

    Question:
    {question}

    Answer:
    """

    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,)
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=32)
    response = tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)

    print(time.time() - start)
    