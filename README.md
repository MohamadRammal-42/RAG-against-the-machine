*This project has been created as part of the 42 curriculum by \<mrammal\>*

# RAG against the machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) pipeline.

The goal of this project is to build a system capable of answering questions using information retrieved from a document collection. Instead of relying only on the knowledge stored inside a language model, the system retrieves relevant external information and uses it as context to generate more accurate and grounded answers.

The project provides a complete RAG workflow:

- Document ingestion and preprocessing.
- Document chunking and metadata extraction.
- Text embedding generation.
- Hybrid document retrieval.
- Retrieval ranking and filtering.
- Answer generation using a language model.
- Retrieval performance evaluation.

The main objective is to combine traditional information retrieval techniques with modern language models in order to create a reliable question-answering system.

---

## Instructions


## Requirements

The project requires:

- Python version: `3.10+`

- PyTorch
- Transformers
- Sentence Transformers
- BM25S
- NumPy
- Fire

Install dependencies:

```bash
uv sync
```

Checking flake8 and mypy:

```bash
make lint
```

Run a default question:

```bash
make run
```

Directly index using the makefile:

```bash
make index
```

Runnig the moulinette "For testing actually" :

```bash
make moul
```

Clean:
```bash
make clean
```

---

# Resources

## Documentation

- Hugging Face Transformers:

<https://huggingface.co/docs/transformers>

- Sentence Transformers:

<https://www.sbert.net/>

- PyTorch Documentation:

<https://pytorch.org/docs/>

- NumPy Documentation:

<https://numpy.org/doc/>

- BM25 Information Retrieval:

<https://en.wikipedia.org/wiki/Okapi_BM25>

---

## Articles and Tutorials

- Retrieval-Augmented Generation:

<https://arxiv.org/abs/2005.11401>

- Dense Passage Retrieval:

<https://arxiv.org/abs/2004.04906>

- Reciprocal Rank Fusion:

<https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>

---

# AI Usage

AI tools were used as assistance during the development of this project.

AI was used for:

- Understanding RAG architecture concepts.
- Researching retrieval techniques.
- Improving documentation structure.
- Reviewing implementation choices.
- Explaining algorithms such as BM25, embeddings, and Reciprocal Rank Fusion.
- Assisting with writing and improving the README documentation.


## System architecture

The RAG pipeline is composed of several interconnected components.

Using the index function:

- Document Loader from the `data/raw/vllm`
- Chunking System

Using the search and search_dataset function:

- `BM25 Index Retrieval` and `Embeddings Generation` Indexing
- Hybrid Search and RRF Ranking the best k chunks

Using answer and answer_dataset:
- Context Builder
- Language Model Answering
- Answer

## Chunking strategy

Documents are loaded from the `vllm`

The documents are processed and transformed into smaller chunks using `PythonCodeTextSplitter` and `MarkdownTextSplitter` from `langchain_text_splitters`.

Each chunk contains metadata:

- Source file path.
- Character start index.
- Character end index.
- Unique chunk identifier.

This metadata allows the system to identify where retrieved information comes from and enables retrieval evaluation.


The goal of chunking is to balance:

- Having enough context for answer generation.
- Keeping chunks small enough for accurate retrieval.

Smaller chunks improve retrieval precision, while larger chunks provide more context.

The selected chunking strategy was chosen because:

- It allows precise source tracking.
- It improves retrieval granularity.
- It makes evaluation possible using character positions.

## Retrieval method

### Embedding Generation

As a bonus, the project uses the following embedding model:

```
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model transforms documents and user queries into numerical vectors.

These vectors allow the system to measure semantic similarity between a question and document chunks.


### BM25 Retrieval

The pipeline uses BM25 as a lexical retrieval method.

BM25 searches documents based on keyword relevance.

Advantages:

- Good performance with exact terminology.
- Efficient retrieval.
- Works well for technical documentation.

## Performance analysis

The system performance is evaluated using recall@k.

The evaluation process:

1. Retrieve documents for each question.
2. Compare retrieved sources with expected sources.
3. Calculate overlap using Intersection over Union (IoU).
4. Determine whether retrieval was successful.

The main metric used is:

```
recall@5
```

A retrieved source is considered correct when:

```
IoU >= 0.05
```

Results:

```
recall@5: <result>
```

The retrieval performance depends on:

- Chunk size.
- Number of retrieved documents (`k`).
- Embedding model quality.
- BM25 indexing.
- Dataset difficulty.

## Design decisions

### Hybrid Retrieval

A combination of BM25 and embeddings was chosen because each method has different strengths.

BM25:

- Good at exact keyword matching.
- Effective for technical terms.

Embeddings:

- Good at semantic understanding.
- Handles different wording.

Combining both improves retrieval coverage.

### Reciprocal Rank Fusion

RRF was selected because:

- It does not require training.
- It combines different ranking systems easily.
- It improves ranking stability.

### Cached Results

The system stores previous:

- Search results.
- Generated answers.

This reduces unnecessary computation when the same question is processed multiple times.

### Stored Embeddings

Generated embeddings are saved to disk.

Benefits:

- Faster future searches.
- Avoids repeating expensive embedding computation.

## Challenges faced

Increasing the recall :)

## Example usage

command: uv run python3 -m src answer "What flag must be explicitly passed when serving VLM2Vec-Full model in vLLM to run it in embedding mode?"

answer: The model must be explicitly passed `--runner pooling` when serving VLM2Vec-Full.