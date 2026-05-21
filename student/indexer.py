import json
import os
from pathlib import Path
from typing import List

import bm25s
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

CODE_EXTENSIONS = ".py"
TEXT_EXTENSIONS = ".md"


def chunk_code_file(text, max_chunk_size) -> List[dict]:
    """Chunk code files."""
    return _chunk_with_text_splitter(text, max_chunk_size)


def _chunk_with_text_splitter(text: str, max_chunk_size: int) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    chunks: List[dict] = []

    for document in splitter.create_documents([text]):
        start_index = int(document.metadata.get("start_index", 0))
        content = document.page_content
        chunks.append(
            {
                "content": content,
                "first_character_index": start_index,
                "last_character_index": start_index + len(content),
            }
        )

    return chunks


def chunk_text_file(text: str, max_chunk_size: int) -> List[dict]:
    """Chunk Markdown files."""
    return _chunk_with_text_splitter(text, max_chunk_size)


def chunk_file_content(
    file_path: Path,
    content: str,
    max_chunk_size: int,
) -> List[dict]:
    """Route file content to the appropriate chunking strategy."""
    if file_path.suffix in CODE_EXTENSIONS:
        return chunk_code_file(content, max_chunk_size)
    if file_path.suffix in TEXT_EXTENSIONS:
        return chunk_text_file(content, max_chunk_size)
    return []


def index_repository(
    repo_path="data/raw/vllm-0.10.1",
    index_dir="data/processed",
    max_chunk_size=2000,
) -> None:
    repo = Path(repo_path)
    output_dir = Path(index_dir)
    print(f"Indexing repository at {repo}...")

    # 1. Walk through files
    files_to_index: List[Path] = []
    for root, _, files in os.walk(repo):
        for file in files:
            if file.endswith(CODE_EXTENSIONS) or file.endswith(TEXT_EXTENSIONS):
                files_to_index.append(Path(root) / file)

    print(f"Found {len(files_to_index)} files. Chunking...")

    # 2. Extract chunks
    all_chunks: List[dict] = []
    for file_path in tqdm(files_to_index):
        try:
            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                content = f.read()

            relative_path = os.path.relpath(file_path, repo)
            file_chunks = chunk_file_content(file_path, content, max_chunk_size)

            for chunk in file_chunks:
                chunk["file_path"] = relative_path
                all_chunks.append(chunk)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not all_chunks:
        print("No chunks found to index.")
        return

    print(f"Total chunks created: {len(all_chunks)}. Building BM25 index")

    # 3. Create BM25 index
    corpus: List = [c["content"] for c in all_chunks]

    # Simple tokenization for BM25S
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # 4. Save index and metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(output_dir / "bm25_index", corpus=corpus)

    # Save chunk metadata for retrieval mapping
    with open(output_dir / "chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Indexing complete. Index saved to {output_dir}")
