import bm25s
from src.splitter import (
    get_files,
    extract_chunks,
    get_chunks,
    extract_paragraphs,
    text_filter)
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from src.message import prepare_message
from .models import (MinimalSource,
                     MinimalSearchResults,
                     MinimalAnswer,
                     StudentSearchResults,
                     StudentSearchResultsAndAnswer,
                     UnansweredQuestion,
                     AnsweredQuestion,
                     RagDataset)
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List
import json
import os


def create_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Create and load the causal language model and tokenizer.

    Loads the Qwen3-0.6B model and its corresponding tokenizer from
    Hugging Face. The model is automatically mapped to available devices.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizerBase]:
            A tuple containing the initialized model and tokenizer.

    Raises:
        RuntimeError:
            If the model or tokenizer cannot be loaded.
    """
    model_id = "Qwen/Qwen3-0.6B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto"
        )
    except OSError as e:
        raise RuntimeError(
            f"Could not load model '{model_id}'."
        ) from e
    return model, tokenizer


def fill_the_source(data: list[Dict[str, Any]]) -> list[MinimalSource]:
    """Convert chunk metadata into minimal source objects.

    Extracts source location information from processed document chunks
    and converts them into ``MinimalSource`` instances.

    Args:
        data (list[Dict[str, Any]]):
            List of processed documents containing chunk metadata.

    Returns:
        list[MinimalSource]:
            A list of source objects containing file paths and character
            offsets.
    """
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
    """Pipeline for document indexing, retrieval, answering, and evaluation.

    This class implements a retrieval-augmented generation (RAG) workflow.
    It combines BM25 lexical retrieval and embedding-based semantic retrieval,
    merges results using reciprocal rank fusion, and optionally generates
    answers using a causal language model.

    Attributes:
        embedder:
            Sentence transformer model used for semantic embeddings.
        retriever:
            BM25 retriever used for lexical search.
        Data (list[Dict[str, Any]]):
            Loaded document data and chunk metadata.
        List_texts (list[str]):
            Text content of all indexed chunks.
        chunk_id (dict[int, Dict[str, Any]]):
            Mapping between chunk identifiers and chunk metadata.
        embeddings (np.ndarray):
            Stored vector representations of document chunks.
        model:
            Language model used for answer generation.
        tokenizer:
            Tokenizer corresponding to the language model.
        cache (Dict[str, Any]):
            Cache of generated answers.
        cache_index (Dict[str, Any]):
            Cache of search results.
        k (int):
            Number of retrieval results returned.
    """
    def __init__(self) -> None:
        """Initialize the RAG pipeline and load required models.

        Initializes the embedding model, BM25 retriever, language model,
        tokenizer, internal caches, and storage containers used throughout
        the retrieval and generation workflow.
        """
        self.embedder: Any = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
            )
        self.retriever = bm25s.BM25()
        self.Data: list[Dict[str, Any]] = []
        self.List_texts: list[str] = []
        self.chunk_id: dict[int, Dict[str, Any]] = {}
        self.chunk_obj: list[Dict[str, Any]] = []
        self.all_units: list[Dict[str, Any]] = []
        self.source: list[MinimalSource] = []
        self.embeddings: np.ndarray = np.array([])
        self.units: list[str] | None = None
        self.model: Any
        self.cache: Dict[str, Any] = {}
        self.cache_index: Dict[str, Any] = {}
        self.tokenizer: PreTrainedTokenizerBase
        self.model, self.tokenizer = create_model()
        self.filtered_text: str | None = None
        self.filterd_chunks: list[Dict[str, Any]] = []
        self.question: UnansweredQuestion = UnansweredQuestion(question="")
        self.msr: MinimalSearchResults = MinimalSearchResults(
            question_id="",
            question="",
            retrieved_sources=[]
        )
        self.Ssr: StudentSearchResults = StudentSearchResults(
            search_results=[], k=0
            )
        self.Ssrna = StudentSearchResultsAndAnswer(
            search_results=[],
            k=0
        )
        self.k: int = 10
        self.Rag_dataset: RagDataset | None = None

    def _load_or_build_embeddings(self, embeddings_path: Path) -> Any:
        """Load cached embeddings if valid, otherwise recompute and cache them.

        Validates that the cached embeddings on disk have the same number of
        rows as ``self.List_texts``. If they don't match (stale cache from a
        previous indexing run), embeddings are recomputed and the cache file
        is overwritten.

        Args:
            embeddings_path (Path):
                Path to the cached embeddings ``.npy`` file.

        Returns:
            np.ndarray:
                Embeddings aligned 1:1 with ``self.List_texts``.
        """
        if not embeddings_path.exists():
            embeddings = self.embedder.encode(
                self.List_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings)
            return embeddings
        cached = np.load(embeddings_path)
        if cached.shape[0] == len(self.List_texts):
            return cached

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the document collection for retrieval.

        Loads raw documents, splits them into chunks, generates embeddings,
        builds a BM25 index, and stores processed artifacts required for
        later retrieval.

        Args:
            max_chunk_size (int, optional):
                Maximum number of characters allowed per chunk. Defaults to
                2000.

        Raises:
            TypeError:
                If ``max_chunk_size`` is not an integer.
            ValueError:
                If ``max_chunk_size`` is less than 1.
            FileNotFoundError:
                If the raw document directory cannot be found.
        """
        if not isinstance(max_chunk_size, int):
            raise TypeError("max_chunk_size must be an integer")
        if max_chunk_size < 1:
            raise ValueError(
                "max_chunk_size must be greater than 0"
                )
        try:
            self.Data, self.List_texts, self.units, self.chunk_id = get_files(
                "data/raw/vllm-0.10.1", max_chunk_size
                )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Raw data directory not found."
            ) from e
        self.chunk_obj = get_chunks(self.Data)
        self.all_units = extract_paragraphs(self.chunk_obj)
        self.source = fill_the_source(self.Data)
        embeddings_path = Path("data/processed/embeddings.npy")
        self.embeddings = self._load_or_build_embeddings(embeddings_path)
        tokens = bm25s.tokenize(self.List_texts)
        self.retriever.index(tokens)
        self.retriever.save("data/processed/bm25_index")
        print("Ingestion complete! Indices saved under data/processed/")

    def search(
            self,
            question: str,
            k: int = 10,
            is_printing: bool = True
            ) -> Any:
        """Generate an answer for a user question using retrieved context.

        Retrieves relevant document chunks, prepares a prompt, generates an
        answer using the language model, and prints the generated response.

        Args:
            question (str):
                User question to answer.
            k (int, optional):
                Number of retrieved chunks used as context. Defaults to 10.

        Raises:
            TypeError:
                If ``k`` is not an integer.
            ValueError:
                If ``k`` is less than 1.
            FileNotFoundError:
                If retrieval indexes are unavailable.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.question = UnansweredQuestion(question=question)
        try:
            with open(
                "data/processed/chunks.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.Data = json.load(f)
            with open(
                "data/processed/index.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.List_texts = json.load(f)
            with open(
                "data/processed/chunk_id.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.chunk_id = {
                    int(k): v
                    for k, v in json.load(f).items()
                }
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Run index() first."
            ) from e
        self.chunk_obj = get_chunks(self.Data)
        self.all_units = extract_paragraphs(self.chunk_obj)
        self.source = fill_the_source(self.Data)
        self.embeddings = self._load_or_build_embeddings(
            Path("data/processed/embeddings.npy")
        )
        self.retriever = bm25s.BM25.load("data/processed/bm25_index")

        def _bm25_retreiving() -> list[Dict[str, Any]]:
            question_tokens = bm25s.tokenize(self.question.question)
            num = min(max(3 * self.k, 20), len(self.List_texts))
            results, scores = self.retriever.retrieve(question_tokens, k=num)
            candidates = [self.List_texts[idx] for idx in results[0]]
            candidates_chunk = extract_chunks(self.Data, candidates)
            ranked = list(zip(candidates_chunk, scores[0]))
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in ranked[:num]]

        def _embed_retreiver() -> list[Dict[str, Any]]:
            ambedding_query = self.embedder.encode(
                self.question.question, normalize_embeddings=True
            )
            num = min(max(3 * self.k, 20), len(self.List_texts))
            scores = self.embeddings @ ambedding_query
            top_idx = np.argsort(scores)[::-1][:num]
            return [self.chunk_id[int(i)] for i in top_idx]

        def _rrf_calculation(
                        ) -> tuple[list[Dict[str, Any]], list[MinimalSource]]:
            bm_chunks = _bm25_retreiving()
            embedding_chunks = _embed_retreiver()
            rrf_ranked: dict[int, float] = {}
            rrf_cst = 60
            for rank, chunk in enumerate(bm_chunks):
                temp = rrf_ranked.get(chunk["chunk_id"], 0) + 1
                rrf_ranked[chunk["chunk_id"]] = temp / (rrf_cst + rank)
            for rank, chunk in enumerate(embedding_chunks):
                temp = rrf_ranked.get(chunk["chunk_id"], 0) + 1
                rrf_ranked[chunk["chunk_id"]] = temp / (rrf_cst * 6 + rank)
            sorted_rrf = sorted(
                rrf_ranked.items(), key=lambda x: x[1], reverse=True
                )[:self.k]
            ranked_chunk_ids = [chunk_id for chunk_id, _ in sorted_rrf]
            filterd_chunks: list[Dict[str, Any]] = []
            ordered_sources: list[MinimalSource] = []
            for chunk in self.chunk_obj:
                if chunk["chunk_id"] in ranked_chunk_ids:
                    filterd_chunks.append(chunk)
                    fi = chunk["first_character_index"]
                    li = chunk["last_character_index"]
                    ordered_sources.append(
                        MinimalSource(
                            file_path=chunk["path"],
                            first_character_index=fi,
                            last_character_index=li,
                        )
                    )
            return filterd_chunks, ordered_sources

        self.filterd_chunks, ListofMinimalSource = _rrf_calculation()
        self.msr = MinimalSearchResults(
            question_id=self.question.question_id,
            question=self.question.question,
            retrieved_sources=ListofMinimalSource,
        )
        if is_printing:
            print("The question was searched")
            return self.filterd_chunks

    def answer(self, question: str, k: int = 10) -> None:
        """Generate an answer for a user question using retrieved context.

        Retrieves relevant document chunks, prepares a prompt, generates an
        answer using the language model, and prints the generated response.

        Args:
            question (str):
                User question to answer.
            k (int, optional):
                Number of retrieved chunks used as context. Defaults to 10.

        Raises:
            TypeError:
                If ``k`` is not an integer.
            ValueError:
                If ``k`` is less than 1.
            FileNotFoundError:
                If retrieval indexes are unavailable.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.search(question, k, False)
        messages = prepare_message(
            self.question.question,
            text_filter(self.filterd_chunks, self.k)
            )
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
            )
        model_inputs = self.tokenizer(text, return_tensors="pt")
        model_inputs = model_inputs.to(self.model.device)
        self.model.eval()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=64
                )
        input_ids = model_inputs["input_ids"]
        input_len = input_ids.shape[1]
        decoded = self.tokenizer.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True
            )
        response = str(decoded)
        answer = AnsweredQuestion(
            question_id=self.msr.question_id,
            question=self.msr.question,
            answer=response,
            sources=self.msr.retrieved_sources
            )
        print(answer.answer)

    def search_dataset(
            self,
            dataset_path: str,
            save_directory: str,
            k: int = 10,
            is_print: bool = True
            ) -> None:
        """Run retrieval evaluation over a question dataset.

        Reads a dataset containing RAG questions, retrieves relevant sources
        for each question, stores the results, and optionally writes them to
        disk.

        Args:
            dataset_path (str):
                Path to the dataset JSON file containing ``rag_questions``.
            save_directory (str):
                Directory where retrieval results are saved.
            k (int, optional):
                Number of retrieved sources per question. Defaults to 10.
            is_print (bool, optional):
                Whether to save and exit after processing. Defaults to True.

        Raises:
            TypeError:
                If ``k`` is not an integer.
            ValueError:
                If ``k`` is less than 1 or the dataset format is invalid.
            FileNotFoundError:
                If the dataset or indexing artifacts are missing.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError("k must be greater than 0")
        self.k = k
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}"
            )
        try:
            with open(
                "data/processed/chunks.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.Data = json.load(f)
            with open(
                "data/processed/index.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.List_texts = json.load(f)
            with open(
                "data/processed/chunk_id.json",
                "r",
                encoding="utf-8",
            ) as f:
                self.chunk_id = {
                    int(k): v
                    for k, v in json.load(f).items()
                }
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Run index() first."
            ) from e
        self.chunk_obj = get_chunks(self.Data)
        self.all_units = extract_paragraphs(self.chunk_obj)
        self.source = fill_the_source(self.Data)
        self.embeddings = self._load_or_build_embeddings(
            Path("data/processed/embeddings.npy")
        )
        self.retriever = bm25s.BM25.load("data/processed/bm25_index")

        def _bm25_retreiving() -> list[Dict[str, Any]]:
            question_tokens = bm25s.tokenize(self.question.question)
            num = min(max(3 * self.k, 20), len(self.List_texts))
            results, scores = self.retriever.retrieve(question_tokens, k=num)
            candidates = [self.List_texts[idx] for idx in results[0]]
            candidates_chunk = extract_chunks(self.Data, candidates)

            ranked = list(zip(candidates_chunk, scores[0]))
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in ranked[:num]]

        def _embed_retreiver() -> list[Dict[str, Any]]:
            ambedding_query = self.embedder.encode(
                self.question.question, normalize_embeddings=True
            )
            num = min(max(3 * self.k, 20), len(self.List_texts))
            scores = self.embeddings @ ambedding_query
            top_idx = np.argsort(scores)[::-1][:num]
            return [
                self.chunk_id[int(i)]
                for i in top_idx
                if int(i) in self.chunk_id
            ]

        def _rrf_calculation(
                        ) -> tuple[list[Dict[str, Any]], list[MinimalSource]]:
            bm_chunks = _bm25_retreiving()
            embedding_chunks = _embed_retreiver()
            rrf_ranked: dict[int, float] = {}
            cst = 60
            for rank, chunk in enumerate(bm_chunks):
                temp = rrf_ranked.get(chunk["chunk_id"], 0)
                rrf_ranked[chunk["chunk_id"]] = temp + 1 / (cst + rank)
            for rank, chunk in enumerate(embedding_chunks):
                temp = rrf_ranked.get(chunk["chunk_id"], 0)
                rrf_ranked[chunk["chunk_id"]] = temp + 1 / (cst * 7 + rank)
            sorted_rrf = sorted(
                rrf_ranked.items(), key=lambda x: x[1], reverse=True
                )[:self.k]
            ranked_chunk_ids = [chunk_id for chunk_id, _ in sorted_rrf]
            len_r = len(ranked_chunk_ids)
            filterd_chunks: List[Any] = [None] * len_r
            ordered_sources: List[Any] = [None] * len_r
            for chunk in self.chunk_obj:
                if chunk["chunk_id"] in ranked_chunk_ids:
                    rank_index = ranked_chunk_ids.index(chunk["chunk_id"])
                    filterd_chunks[rank_index] = chunk
                    fi = chunk["first_character_index"]
                    li = chunk["last_character_index"]
                    ordered_sources[rank_index] = MinimalSource(
                        file_path=chunk["path"],
                        first_character_index=fi,
                        last_character_index=li,
                    )
            return filterd_chunks, ordered_sources

        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        ssr = []
        if "rag_questions" not in dataset:
            raise ValueError("Dataset missing 'rag_questions'.")
        for query in tqdm(dataset["rag_questions"]):
            if "question" not in query.keys():
                raise ValueError("Missing question")
            if "question_id" not in query.keys():
                raise ValueError("Missing question_id")
            self.question = UnansweredQuestion(
                question_id=query["question_id"],
                question=query["question"]
                )
            if self.question.question in self.cache_index.keys():
                self.msr = self.cache_index[self.question.question]
            else:
                self.filterd_chunks, ListofMinimalSource = _rrf_calculation()
                self.msr = MinimalSearchResults(
                    question_id=self.question.question_id,
                    question=self.question.question,
                    retrieved_sources=ListofMinimalSource,
                )
                self.cache_index[self.question.question] = self.msr
            ssr.append(self.msr)
        self.Ssr = StudentSearchResults(search_results=ssr, k=self.k)
        if is_print:
            output_file = Path(
                f"{save_directory}/{os.path.basename(dataset_path)}"
                )
            print(f"Saved student_search_results to {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            search_result_data = self.Ssr.model_dump()
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(search_result_data, f, indent=4)
            sys.exit(0)

    def answer_dataset(
            self,
            dataset_path: str,
            save_directory: str,
            k: int = 10
            ) -> None:
        """Generate answers for all questions in a dataset.

        Performs retrieval for each question, generates model responses using
        retrieved context, caches repeated questions, and saves the resulting
        answers.

        Args:
            dataset_path (str):
                Path to the dataset JSON file containing questions.
            save_directory (str):
                Directory where generated results are stored.
            k (int, optional):
                Number of retrieved chunks used for generation. Defaults
                to 10.

        Raises:
            TypeError:
                If ``k`` is not an integer.
            ValueError:
                If ``k`` is less than 1.
            FileNotFoundError:
                If the dataset or required indexes cannot be found.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError("k must be greater than 0")
        self.k = k
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        self.search_dataset(dataset_path, save_directory, self.k, False)
        List_ma = []
        rag_questions: List[AnsweredQuestion | UnansweredQuestion] = []
        count = 0
        for s in self.Ssr.search_results:
            count += 1
            self.question = UnansweredQuestion(
                question=s.question,
                question_id=s.question_id
                )
            if self.question.question in self.cache.keys():
                ma = self.cache[self.question.question]
            else:
                messages = prepare_message(
                    self.question.question,
                    text_filter(self.filterd_chunks, self.k)
                    )
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                    )
                model_inputs = self.tokenizer(text, return_tensors="pt")
                model_inputs = model_inputs.to(self.model.device)
                self.model.eval()
                with torch.inference_mode():
                    generated_ids: Any = self.model.generate(
                        **model_inputs,
                        max_new_tokens=64
                        )
                input_ids = model_inputs["input_ids"]
                input_len = input_ids.shape[1]
                decoded = self.tokenizer.decode(
                    generated_ids[0][input_len:],
                    skip_special_tokens=True
                    )
                response = str(decoded)
                answer = AnsweredQuestion(
                    question_id=self.question.question_id,
                    question=self.question.question,
                    answer=response,
                    sources=s.retrieved_sources
                    )
                ma = MinimalAnswer(
                    question_id=self.question.question_id,
                    question=self.question.question,
                    retrieved_sources=s.retrieved_sources,
                    answer=answer.answer
                    )
                self.cache[self.question.question] = ma
            rag_questions.append(answer)
            List_ma.append(ma)
        self.Ssrna = StudentSearchResultsAndAnswer(
            search_results=List_ma,
            k=self.k
            )
        self.RagDataset = RagDataset(
            rag_questions=rag_questions
        )
        output_file = Path(
            f"{save_directory}/{os.path.basename(dataset_path)}"
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        search_and_answer_result_data = self.Ssrna.model_dump()
        print(
            f"Loaded {count} questions ..."
            f"Processed {count} of {count} questions"
            )
        print(f"Saved student_search_results_and_answer to {output_file}")
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(search_and_answer_result_data, f, indent=4)
        sys.exit(0)

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str
            ) -> None:
        """Evaluate retrieval performance against a reference dataset.

        Computes recall@5 by measuring overlap between retrieved source
        locations and expected source locations using intersection over union
        (IoU).

        Args:
            student_search_results_path (str):
                Path to generated retrieval results.
            dataset_path (str):
                Path to the reference dataset containing expected sources.

        Raises:
            FileNotFoundError:
                If either input file does not exist.
            ValueError:
                If the dataset structure is invalid.
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        student_search_results_file = Path(student_search_results_path)
        if not student_search_results_file.is_file():
            raise FileNotFoundError(
                f"Result file not found: {student_search_results_path}"
            )
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
            intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
            union = max(a_end, b_end) - min(a_start, b_start)
            return intersection / union if union > 0 else 0.0

        if "rag_questions" not in dataset:
            raise ValueError("Dataset missing 'rag_questions'.")
        num_of_overlapp = 0
        for i, query in enumerate(dataset["rag_questions"]):
            if "question" not in query:
                raise ValueError(f"Question {i} missing 'question'.")
            if "question_id" not in query:
                raise ValueError(f"Question {i} missing 'question_id'.")
            if "sources" not in query:
                raise ValueError(f"Question {i} missing 'sources'.")
            if not isinstance(query["sources"], list):
                raise ValueError(f"Question {i} 'sources' must be a list.")
            for j, source in enumerate(query["sources"]):
                if "file_path" not in source:
                    raise ValueError(
                        f"Question {i}, source {j} missing 'file_path'."
                    )
                if "first_character_index" not in source:
                    raise ValueError(
                        f"Question {i}, source {j} missing"
                        f"'first_character_index'."
                    )
                if "last_character_index" not in source:
                    raise ValueError(
                        f"Question {i}, source {j} missing"
                        f"'last_character_index'."
                    )

            def checking_overlap() -> float:
                for sec_query in results["search_results"]:
                    if sec_query["question_id"] != query["question_id"]:
                        continue
                    for retrieved in sec_query["retrieved_sources"][:5]:
                        for source in query["sources"]:
                            if retrieved["file_path"] != source["file_path"]:
                                continue
                            iou = _iou(
                                source["first_character_index"],
                                source["last_character_index"],
                                retrieved["first_character_index"],
                                retrieved["last_character_index"],
                            )
                            if iou >= 0.05:
                                return iou
                    return 0.0
                return 0.0

            if checking_overlap() >= 0.05:
                num_of_overlapp += 1
        i = i + 1
        print(f"recall@5: {(num_of_overlapp / i) * 100:.1f}%")
        sys.exit(0)
