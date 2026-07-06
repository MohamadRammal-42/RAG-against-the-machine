import bm25s
from src.splitter import get_files, extract_chunks, get_chunks, extract_paragraphs, text_filter
from sentence_transformers import CrossEncoder, SentenceTransformer
from functools import lru_cache
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.message import prepare_message
from .models import MinimalSource, MinimalSearchResults, MinimalAnswer, StudentSearchResults, StudentSearchResultsAndAnswer, UnansweredQuestion, AnsweredQuestion, RagDataset
import torch
from pathlib import Path
import time
import json
import os


def create_model():
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )
    return model, tokenizer

def fill_the_source(data):
    sources = []
    for d in data:
        for chunk in d["content"].values():
            source = MinimalSource(
                file_path=d["path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            )
            sources.append(source)
    return sources


class RAGPipeline:
    def __init__(self) -> None:
        self.Data = None
        self.List_texts = None
        self.units = None
        self.chunk_id = None
        self.chunk_obj = None
        self.all_units = None
        self.source = None
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.model = None
        self.retriever = bm25s.BM25()
        self.embeddings = None
        self.filtered_text = None
        self.filterd_chunks = None
        self.tokenizer = None
        self.question = None
        self.msr = None
        self.Ssr = None
        self.Ssrna = None
        self.k = 5


    def index(self, max_chunk_size: int = 2000):
        self.Data, self.List_texts, self.units, self.chunk_id = get_files("data/raw/vllm-0.10.1", max_chunk_size)
        self.chunk_obj = get_chunks(self.Data)
        self.all_units = extract_paragraphs(self.chunk_obj)
        self.source = fill_the_source(self.Data)
        embeddings_path = Path("data/processed/embeddings.npy")

        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        else:
            self.embeddings = self.embedder.encode(
                self.List_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, self.embeddings)

        tokens = bm25s.tokenize(self.List_texts)
        self.retriever.index(tokens)
        self.retriever.save("data/processed/bm25_index")
        print("Ingestion complete! Indices saved under data/processed/")


    def search(self, question: str, k: int = 5, is_printing: bool = True):
        self.k = k
        self.question = UnansweredQuestion(question=question)
        with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
            self.Data = json.load(f)
        with open("data/processed/index.json", "r", encoding="utf-8") as f:
            self.List_texts = json.load(f)
        with open("data/processed/chunk_id.json", "r", encoding="utf-8") as f:
            self.chunk_id = {int(k): v for k, v in json.load(f).items()}
        self.chunk_obj = get_chunks(self.Data)
        self.all_units = extract_paragraphs(self.chunk_obj)
        self.source = fill_the_source(self.Data)
        self.embeddings = np.load("data/processed/embeddings.npy")
        self.retriever = bm25s.BM25.load("data/processed/bm25_index")
        @lru_cache(None)
        def _bm25_retreiving():
            question_tokens = bm25s.tokenize(self.question.question)
            num = min(max(3 * self.k, 20), len(self.List_texts))
            results, scores = self.retriever.retrieve(question_tokens, k=num)
            candidates = [self.List_texts[idx] for idx in results[0]]
            candidates_chunk = extract_chunks(self.Data, candidates)

            ranked = list(zip(candidates_chunk, scores[0]))
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in ranked[:num]]

        @lru_cache(None)
        def _embed_retreiver():
            ambedding_query = self.embedder.encode(
                self.question.question, normalize_embeddings=True
            )
            num = min(max(3 * self.k, 20), len(self.List_texts))
            scores = self.embeddings @ ambedding_query
            top_idx = np.argsort(scores)[::-1][:num]
            return [self.chunk_id[int(i)] for i in top_idx]

        def _rrf_calculation(k):
            self.k = k
            num = min(max(3 * self.k, 20), len(self.List_texts))
            # reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            # bm_chunks = _bm25_retreiving(question)
            # embedding_chunks = _embed_retreiver(question)

            # filtered_chunk = []
            # for i in range(num):
            #     filtered_chunk.append(bm_chunks[i])
            # filtered_chunk = bm_chunks[:num]
            # filtered_chunk.extend(embedding_chunks[:num])
            # chunks = [chunk["text"] for chunk in filtered_chunk]
            # paires = [(question, chunk) for chunk in chunks]
            # scores = reranker.predict(paires)
            # ranked = [c for _, c in sorted(zip(scores, chunks), reverse=True)]

            # return list(set(ranked))[:self.k]
            bm_chunks = _bm25_retreiving()
            embedding_chunks = _embed_retreiver()

            rrf_ranked = {}

            for rank, chunk in enumerate(bm_chunks):
                rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k + rank)
            for rank, chunk in enumerate(embedding_chunks):
                rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k * 3 + rank)

            rrf_ranked = sorted(rrf_ranked.items(), key=lambda x: x[1], reverse=True)[:k]
            filterd_chunks = [0] * k
            rrf_ranked = [chunk for chunk, _ in rrf_ranked]
            ListofMinimalSource = []
            for chunk in self.chunk_obj:
                if chunk["chunk_id"] in rrf_ranked:
                    filterd_chunks[rrf_ranked.index(chunk["chunk_id"])] = chunk
                    ListofMinimalSource.append(MinimalSource(
                        file_path=chunk["path"],
                        first_character_index=chunk["first_character_index"],
                        last_character_index=chunk["last_character_index"],
                    ))

            return filterd_chunks, ListofMinimalSource

        self.filterd_chunks, ListofMinimalSource = _rrf_calculation(self.k)
        self.msr = MinimalSearchResults(
            question_id=self.question.question_id,
            question=self.question.question,
            retrieved_sources=ListofMinimalSource,
        )


    def answer(self, question: str, k: int = 5):
        self.k = k
        self.search(question, k, False)
        self.model, self.tokenizer = create_model()

        messages = prepare_message(self.question.question, self.filterd_chunks[:5])
        start = time.time()
        print("start")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        self.model.eval()
        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=64)
        response = self.tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answer = AnsweredQuestion(question_id=self.msr.question_id, question=self.msr.question, answer=response, sources=self.msr.retrieved_sources)
        print(time.time() - start)
        return answer


    def search_dataset(self, dataset_path, save_directory, k: int = 5, is_print: bool = True):
        self.k = k
        with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
            self.Data = json.load(f)
        with open("data/processed/index.json", "r", encoding="utf-8") as f:
            self.List_texts = json.load(f)
        with open("data/processed/chunk_id.json", "r", encoding="utf-8") as f:
            self.chunk_id = {int(k): v for k, v in json.load(f).items()}
        self.chunk_obj = get_chunks(self.Data)
        self.all_units = extract_paragraphs(self.chunk_obj)
        self.source = fill_the_source(self.Data)
        self.embeddings = np.load("data/processed/embeddings.npy")
        self.retriever = bm25s.BM25.load("data/processed/bm25_index")

        def _bm25_retreiving():
            question_tokens = bm25s.tokenize(self.question.question)
            num = min(max(3 * self.k, 20), len(self.List_texts))
            results, scores = self.retriever.retrieve(question_tokens, k=num)
            candidates = [self.List_texts[idx] for idx in results[0]]
            candidates_chunk = extract_chunks(self.Data, candidates)

            ranked = list(zip(candidates_chunk, scores[0]))
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in ranked[:num]]

        def _embed_retreiver():
            ambedding_query = self.embedder.encode(
                self.question.question, normalize_embeddings=True
            )
            num = min(max(3 * self.k, 20), len(self.List_texts))
            scores = self.embeddings @ ambedding_query
            top_idx = np.argsort(scores)[::-1][:num]
            return [self.chunk_id[int(i)] for i in top_idx]

        def _rrf_calculation(k):
            self.k = k
            num = min(max(3 * self.k, 20), len(self.List_texts))
            bm_chunks = _bm25_retreiving()
            embedding_chunks = _embed_retreiver()

            rrf_ranked = {}

            for rank, chunk in enumerate(bm_chunks):
                rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k + rank)
            for rank, chunk in enumerate(embedding_chunks):
                rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k * 3 + rank)

            rrf_ranked = sorted(rrf_ranked.items(), key=lambda x: x[1], reverse=True)[:k]
            filterd_chunks = [0] * k
            rrf_ranked = [chunk for chunk, _ in rrf_ranked]
            ListofMinimalSource = []
            for chunk in self.chunk_obj:
                if chunk["chunk_id"] in rrf_ranked:
                    filterd_chunks[rrf_ranked.index(chunk["chunk_id"])] = chunk
                    ListofMinimalSource.append(MinimalSource(
                        file_path=chunk["path"],
                        first_character_index=chunk["first_character_index"],
                        last_character_index=chunk["last_character_index"],
                    ))
            return filterd_chunks, ListofMinimalSource

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        ssr = []
        for query in dataset["rag_questions"]:
            self.question = UnansweredQuestion(question_id = query["question_id"], question=query["question"])
            self.filterd_chunks, ListofMinimalSource = _rrf_calculation(self.k)
            self.msr = MinimalSearchResults(
                question_id=self.question.question_id,
                question=self.question.question,
                retrieved_sources=ListofMinimalSource,
            )
            ssr.append(self.msr)

        self.Ssr = StudentSearchResults(search_results=ssr, k=self.k)
        if is_print:
            output_file = Path(f"{save_directory}/{os.path.basename(dataset_path)}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            search_result_data = self.Ssr.model_dump()
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(search_result_data, f, indent=4)


    def answer_dataset(self, dataset_path, save_directory, k: int = 5):
        self.k = k
        self.search_dataset(dataset_path, save_directory, self.k, False)
        self.model, self.tokenizer = create_model()

        List_ma = []
        for s in self.Ssr.search_results:
            self.question = UnansweredQuestion(question=s.question, question_id=s.question_id)
            print(self.question)
            messages = prepare_message(self.question.question, self.filterd_chunks[:self.k])
            start = time.time()
            print("start")
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            self.model.eval()
            with torch.inference_mode():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=64)
            response = self.tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            answer = AnsweredQuestion(question_id=self.question.question_id, question=self.question.question, answer=response, sources=s.retrieved_sources)
            ma = MinimalAnswer(question_id=self.question.question_id, question=self.question.question, retrieved_sources=s.retrieved_sources, answer=response)
            List_ma.append(ma)
            print(time.time() - start)
        self.Ssrna = StudentSearchResultsAndAnswer(search_results=List_ma, k=self.k)
        output_file = Path(f"{save_directory}/{os.path.basename(dataset_path)}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        search_and_answer_result_data = self.Ssrna.model_dump()
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(search_and_answer_result_data, f, indent=4)





        # search_result_data = dict()
        # search_results = []
        # for s in self.Ssr.search_results:
        #     search_result = dict()
        #     search_result["question_id"] = s.question_id
        #     search_result["question"] = s.question
        #     retrieved_sources = []
        #     for m in s.retrieved_sources:
        #         source_result = dict()
        #         source_result["file_path"] = m.file_path
        #         source_result["first_character_index"] = m.first_character_index
        #         source_result["last_character_index"] = m.last_character_index
        #         retrieved_sources.append(source_result)
        #     search_result["retrieved_sources"] = retrieved_sources
        #     search_results.append(search_result)

        # search_result_data["search_results"] = search_results
        # search_result_data["k"] = self.k



























def main():
    k = 10
    Data, List_texts, units, chunk_id = get_files("./vllm-0.10.1")
    chunk_obj = get_chunks(Data)
    all_units = extract_paragraphs(chunk_obj)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = embedder.encode(
    #     List_texts, batch_size=64, normalize_embeddings=True,
    #     show_progress_bar=True
    # )
    # np.save("data/processed/embeddings.npy", embeddings)
    embeddings = np.load("data/processed/embeddings.npy")

    tokens = bm25s.tokenize(List_texts)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save("data/processed/bm25_index")

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
        top_idx = np.argsort(scores)[::-1][:num]
        return [chunk_id[i] for i in top_idx]

    def _rrf_calculation(question, k):
        bm_chunks = _bm25_retreiving(question)
        embedding_chunks = _embed_retreiver(question)

        rrf_ranked = {}

        for rank, chunk in enumerate(bm_chunks):
            rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k + rank)
        for rank, chunk in enumerate(embedding_chunks):
            rrf_ranked[chunk["chunk_id"]] = rrf_ranked.get(chunk["chunk_id"], 0) + 1 / (k + 3 * rank)

        rrf_ranked = sorted(rrf_ranked.items(), key=lambda x: x[1], reverse=True)[:k]
        filterd_chunks = [0] * k
        rrf_ranked = [chunk for chunk, _ in rrf_ranked]
        for chunk in chunk_obj:
            if chunk["chunk_id"] in rrf_ranked:
                filterd_chunks[rrf_ranked.index(chunk["chunk_id"])] = chunk

        return filterd_chunks

    question = "What condition prevents tensors from being moved to device in FlashInfer attention backend's prepare method?"
    import time

    filterd_chunks = _rrf_calculation(question, k)
    filtered_text = text_filter(filterd_chunks, k)
    messages = prepare_message(question, filterd_chunks[:5])

    start = time.time()
    print("start")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, max_new_tokens=64)
    response = tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)

    print(time.time() - start)
