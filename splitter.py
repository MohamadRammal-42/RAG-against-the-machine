from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter
)
from pathlib import Path
import os
import json

def mark_down_splitter(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    header_docs = markdown_splitter.split_text(text)
    code_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    final_docs = {}
    final_docs["path"] = path
    final_docs["content"] = {}
    chunk_number = 1
    first_index = 0
    last_index = 0
    for doc in header_docs:
        chunks = code_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunk_data = {}
            chunk_data["text"] = chunk
            chunk_data["metadata"] = doc.metadata
            chunk_data["first_character_index"] = first_index
            last_index += len(chunk) - 1
            chunk_data["last_character_index"] = last_index
            final_docs["content"][f"chunk{chunk_number}"] = chunk_data
            chunk_data["path"] = path
            chunk_number += 1
            first_index = last_index + 1
            last_index = first_index

    return final_docs


def python_splitter(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = PythonCodeTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )
    chunks = splitter.split_text(text)

    final_docs = {}
    final_docs["path"] = path
    final_docs["content"] = {}
    chunk_number = 1
    first_index = 0
    last_index = 0
    for chunk in chunks:
        chunk_data = {}
        chunk_data["text"] = chunk
        chunk_data["first_character_index"] = first_index
        last_index += len(chunk) - 1
        chunk_data["last_character_index"] = last_index
        final_docs["content"][f"chunk{chunk_number}"] = chunk_data
        chunk_data["path"] = path
        chunk_number += 1
        first_index = last_index + 1
        last_index = first_index

    return final_docs

def get_files(root_dir):
    Data = []
    List_texts = []
    paths_for_py = []
    paths_for_md = []

    for file_path in Path(root_dir).rglob("*"):
        if file_path.is_file() and file_path.suffix in {".py", ".md"}:
            if ".md" in str(file_path):
                paths_for_md.append(str(file_path))
            else:
                paths_for_py.append(str(file_path))

    for r in range(len(paths_for_md)):
        data = mark_down_splitter(paths_for_md[r])
        Data.append(data)

    for r in range(len(paths_for_py)):
        data = python_splitter(paths_for_py[r])
        Data.append(data)

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/data.json", "w") as f:
        json.dump(Data, f, indent=4)

    for d in Data:
        for chunk, v in d['content'].items():
            List_texts.append(v['text'])

    with open("data/processed/index.json", "w") as f:
        json.dump(List_texts, f, indent=4)

    embedding_id = 0
    units = []
    id = 0
    chunk_id = {}
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

    with open("data/processed/data.json", "w") as f:
        json.dump(Data, f, indent=4)

    return Data, List_texts, units, chunk_id

def extract_chunks(Data, candidates):
    candidates_chunks = []
    for can in candidates:
        for doc in Data:
            for chunk in doc["content"]:
                if doc["content"][chunk]["text"] == can:
                    candidates_chunks.append(doc["content"][chunk])

    return candidates_chunks

def get_chunks(Data):
    chunk_obj = []
    for d in Data:
        for chunk in d["content"]:
            chunk_obj.append(d["content"][chunk])
    return chunk_obj

def extract_paragraphs(chunk_obj):
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