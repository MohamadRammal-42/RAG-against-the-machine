from langchain_text_splitters import (
    PythonCodeTextSplitter,
    MarkdownTextSplitter
)
from typing import Dict, Any
from pathlib import Path
import os
import json


def mark_down_splitter(path: str, max_chunk_size: int) -> Dict[str, Any]:
    """Split a Markdown document into overlapping chunks.

    Reads a Markdown file, divides it into overlapping text chunks using
    ``MarkdownTextSplitter``, and records metadata describing the location
    of each chunk within the original document.

    Args:
        path (str):
            Path to the Markdown file.
        max_chunk_size (int):
            Maximum size of each generated chunk.

    Returns:
        Dict[str, Any]:
            A dictionary containing the document path and metadata for each
            generated chunk, including its text and character positions.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = MarkdownTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=int(max_chunk_size * 0.35)
    )
    chunks = splitter.split_text(text)

    final_docs: dict[str, Any] = {}
    final_docs["path"] = path
    final_docs["content"] = {}

    chunk_number = 1
    search_start = 0

    for chunk in chunks:
        chunk_data: dict[str, Any] = {}
        chunk_data["text"] = chunk
        chunk_data["path"] = path
        first_index = text.find(chunk, search_start)
        if first_index == -1:
            first_index = search_start
        last_index = first_index + len(chunk) - 1
        chunk_data["first_character_index"] = first_index
        chunk_data["last_character_index"] = last_index
        final_docs["content"][f"chunk{chunk_number}"] = chunk_data
        chunk_number += 1
        search_start = first_index + int(len(chunk) * 0.5)

    return final_docs


def python_splitter(path: str, max_chunk_size: int,) -> Dict[str, Any]:
    """Split a Python source file into overlapping chunks.

    Reads a Python file, divides it into overlapping code chunks using
    ``PythonCodeTextSplitter``, and records metadata describing the location
    of each chunk within the original source file.

    Args:
        path (str):
            Path to the Python source file.
        max_chunk_size (int):
            Maximum size of each generated chunk.

    Returns:
        Dict[str, Any]:
            A dictionary containing the document path and metadata for each
            generated chunk, including its text and character positions.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = PythonCodeTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=int(max_chunk_size * 0.2)
    )
    chunks = splitter.split_text(text)
    final_docs: dict[Any, Any] = {}
    final_docs["path"] = path
    final_docs["content"] = {}
    chunk_number = 1
    search_start = 0

    for chunk in chunks:
        chunk_data: Any = {}
        chunk_data["text"] = chunk
        chunk_data["path"] = path
        first_index = text.find(chunk, search_start)
        if first_index == -1:
            first_index = search_start
        last_index = first_index + len(chunk) - 1
        chunk_data["first_character_index"] = first_index
        chunk_data["last_character_index"] = last_index
        final_docs["content"][f"chunk{chunk_number}"] = chunk_data
        chunk_number += 1
        search_start = first_index + int(len(chunk) * 0.5)

    return final_docs


def get_files(
        root_dir: str,
        max_chunk_size: int
        ) -> (
            tuple[list[Dict[str, Any]],
                  list[str], list[str],
                  Dict[int, Dict[str, Any]]]):
    """Process supported documents and build indexing metadata.

    Recursively scans a directory for Markdown and Python files, splits them
    into chunks, generates metadata required for retrieval, and stores the
    processed data under ``data/processed``.

    Args:
        root_dir (str):
            Root directory containing the documents to process.
        max_chunk_size (int):
            Maximum size of generated chunks.

    Returns:
        tuple[
            list[Dict[str, Any]],
            list[str],
            list[str],
            Dict[int, Dict[str, Any]]
        ]:
            A tuple containing:

            - Processed document metadata.
            - List of chunk texts.
            - List of embedding units.
            - Mapping between embedding IDs and chunk metadata.
    """
    Data: list[dict[str, Any]] = []
    List_texts: list[str] = []
    paths_for_py: list[str] = []
    paths_for_md: list[str] = []

    units: list[str] = []
    chunk_id: dict[int, dict[str, Any]] = {}

    for file_path in Path(root_dir).rglob("*"):
        if file_path.is_file() and file_path.suffix in {".py", ".md"}:
            if ".md" in str(file_path):
                paths_for_md.append(str(file_path))
            else:
                paths_for_py.append(str(file_path))

    for r in range(len(paths_for_md)):
        data = mark_down_splitter(paths_for_md[r], max_chunk_size)
        Data.append(data)

    for r in range(len(paths_for_py)):
        data = python_splitter(paths_for_py[r], max_chunk_size)
        Data.append(data)

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/chunks.json", "w") as f:
        json.dump(Data, f, indent=4)

    for d in Data:
        for chunk, v in d['content'].items():
            List_texts.append(v['text'])

    with open("data/processed/index.json", "w") as f:
        json.dump(List_texts, f, indent=4)

    embedding_id = 0
    id = 0
    for doc in Data:
        for chunk in doc["content"]:
            doc["content"][chunk]["chunk_id"] = id
            id += 1
            if doc["path"].endswith(".md"):
                doc["content"][chunk]["paragraphs"] = list()
                chunk_paragraphs = [
                    p.strip()
                    for p in doc["content"][chunk]["text"].split("\n\n")
                    if p.strip()
                ]
                i = 0
                for p in chunk_paragraphs:
                    doc["content"][chunk]["paragraphs"].append(
                        {
                            "id": i,
                            "embedding_id": embedding_id,
                            "text": p.strip()
                        }
                    )
                    units.append(p)
                    i += 1
                    chunk_id[embedding_id] = doc["content"][chunk]
                    embedding_id += 1
            else:
                doc["content"][chunk]["embedding_id"] = embedding_id
                embedding_id += 1
                chunk_id[embedding_id] = doc["content"][chunk]
                units.append(doc["content"][chunk]["text"])

    with open("data/processed/chunks.json", "w") as f:
        json.dump(Data, f, indent=4)

    with open("data/processed/chunk_id.json", "w") as f:
        json.dump(chunk_id, f, indent=4)

    return Data, List_texts, units, chunk_id


def extract_chunks(
        Data: list[Dict[str, Any]],
        candidates: list[str]
        ) -> list[Dict[str, Any]]:
    """Retrieve chunk metadata for candidate texts.

    Searches the processed document collection and returns the chunk objects
    whose text matches the supplied candidate strings.

    Args:
        Data (list[Dict[str, Any]]):
            Collection of processed documents.
        candidates (list[str]):
            Chunk texts selected by a retrieval method.

    Returns:
        list[Dict[str, Any]]:
            Metadata describing the matching document chunks.
    """
    candidates_chunks: list[dict[str, Any]] = []
    for can in candidates:
        for doc in Data:
            for chunk in doc["content"]:
                if doc["content"][chunk]["text"] == can:
                    candidates_chunks.append(doc["content"][chunk])

    return candidates_chunks


def get_chunks(Data: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Extract all chunks from the processed documents.

    Flattens the processed document structure into a single list of chunk
    dictionaries.

    Args:
        Data (list[Dict[str, Any]]):
            Collection of processed documents.

    Returns:
        list[Dict[str, Any]]:
            List containing every chunk in the dataset.
    """
    chunk_obj: list[dict[str, Any]] = []
    for d in Data:
        for chunk in d["content"]:
            chunk_obj.append(d["content"][chunk])
    return chunk_obj


def extract_paragraphs(
        chunk_obj: list[Dict[str, Any]]
        ) -> list[Dict[str, Any]]:
    """Extract embedding units from document chunks.

    For Markdown chunks, each paragraph is treated as an individual embedding
    unit. For Python chunks, the entire chunk is treated as a single unit.

    Args:
        chunk_obj (list[Dict[str, Any]]):
            List of processed document chunks.

    Returns:
        list[Dict[str, Any]]:
            List of embedding units containing text and embedding IDs.
    """
    all_units = []
    for chunk in chunk_obj:
        parts = chunk.get("paragraphs", None)
        if not parts:
            all_units.append({
                "text": chunk["text"],
                "embedding_id": chunk["embedding_id"]
            })
        else:
            for part in parts:
                all_units.append({
                    "text": part["text"],
                    "embedding_id": part["embedding_id"]
                })
    return all_units


def text_filter(filtered_chunk: list[Dict[str, Any]], k: int) -> str:
    """Concatenate the text of the top retrieved chunks.

    Builds the context passed to the language model by joining the text from
    the first ``k`` retrieved chunks.

    Args:
        filtered_chunk (list[Dict[str, Any]]):
            Ranked list of retrieved chunks.
        k (int):
            Number of chunks to include.

    Returns:
        str:
            Concatenated text of the selected chunks.
    """
    filtered_text = ""
    for chunk in filtered_chunk[:k]:
        filtered_text += chunk["text"]
    return filtered_text
