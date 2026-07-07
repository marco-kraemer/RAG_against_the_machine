import json
import os
from pathlib import Path
from typing import List

import bm25s
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm


def chunk_text(text: str, max_chunk_size: int) -> List[dict]:
    """Chunk files using RecursiveCharacterTextSplitter."""
    chunk_overlap = max_chunk_size // 10  # 10% overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
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


def chunk_code(text: str, max_chunk_size: int) -> List[dict]:
    """Chunk Python source on class/def boundaries (code-aware).

    Uses Python-structural separators so chunks align to real code
    units (classes, functions) instead of arbitrary prose breaks,
    with 50% overlap so a definition is not lost across a boundary.
    """
    chunk_overlap = max_chunk_size // 2  # 50% overlap
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
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


def chunk_file(
    text: str, max_chunk_size: int, file_extension: str
) -> List[dict]:
    """Chunk a file with the strategy matching its type.

    Python files use code-aware splitting; everything else uses the
    prose splitter.

    Args:
        text: Full text content of the file.
        max_chunk_size: Maximum chunk size in characters.
        file_extension: File suffix used to pick the strategy (e.g. ".py").

    Returns:
        The list of chunk dicts produced by the chosen splitter.
    """
    if file_extension == ".py":
        return chunk_code(text, max_chunk_size)
    return chunk_text(text, max_chunk_size)


def index_repository(
    repo_path: str = "data/raw/vllm-0.10.1",
    index_dir: str = "data/processed",
    max_chunk_size: int = 2000,
) -> None:
    """Index a repository into a persisted BM25 store.

    Walks the repository for ``.py`` and ``.md`` files, chunks them with
    the per-type strategy, builds a BM25 index, and writes the index plus
    a ``chunks.json`` metadata file to ``index_dir``.

    Args:
        repo_path: Path to the source repository to index.
        index_dir: Directory where the BM25 index and chunk metadata are
            written.
        max_chunk_size: Maximum chunk size in characters.
    """
    repo = Path(repo_path)
    output_dir = Path(index_dir)
    print(f"Indexing repository at {repo}...")

    # 1. Walk through files
    files_to_index: List[Path] = []
    for root, _, files in os.walk(repo):
        for file in files:
            if file.endswith(".py") or file.endswith(".md"):
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
            file_chunks = chunk_file(content, max_chunk_size, file_path.suffix)

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
